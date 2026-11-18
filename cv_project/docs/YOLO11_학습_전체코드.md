# YOLO11 학습 전체 코드 (Ultralytics 8.4.208 원문, 실행 순서대로)

- 학습(`model.train`)이 실제로 지나가는 함수를 실행 순서대로, 줄임 없이 이어 붙였다(1,210줄). 바꾼 것은 영어 설명문(docstring)과 빈 줄 삭제뿐이다.
- 각 함수 위 `# ──` 줄 = 파일 위치(`.venv/Lib/site-packages/ultralytics/` 기준)와 원본 줄 번호.
- 단계별 해설: claude.ai 문서 'YOLO11 학습 과정 해설' https://claude.ai/artifact/QupsSEAb49zHyrgRCFdjTx (1~9절).
- Ultralytics는 AGPL-3.0. 공부·과제용 인용이다.

## 0-1. 모델 만들기

```python
# ── models/yolo/detect/train.py  DetectionTrainer.get_model  (줄 184~200)
def get_model(self, cfg: str | dict | None = None, weights: torch.nn.Module | None = None, verbose: bool = True):
    model = self.set_model_names_for_load(
        DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
    )
    if weights:
        model.load(weights)
    return model

# ── nn/tasks.py  DetectionModel.__init__  (줄 523~561)
def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
    super().__init__()
    _initialize_yolo_model(self, cfg, ch, nc, verbose)
    # Build strides
    m = self.model[-1]  # Detect()
    if isinstance(m, Detect):  # includes all Detect subclasses like Segment, Pose, OBB, YOLOEDetect, YOLOESegment
        s = 256  # 2x min stride
        m.inplace = self.inplace
        def _forward(x):
            output = self.forward(x)
            if "one2many" in output:
                output = output["one2many"]
            return output["feats"]
        self.model.eval()  # Avoid changing batch statistics until training begins
        m.training = True  # Setting it to True to properly return strides
        m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])  # forward
        self.stride = m.stride
        self.model.train()  # Set model back to training(default) mode
        m.bias_init()  # only run once
    else:
        self.stride = torch.Tensor([32])  # default stride, e.g., RTDETR
    # Init weights, biases
    initialize_weights(self)
    if verbose:
        self.info()
        LOGGER.info("")

# ── nn/tasks.py  BaseModel.load  (줄 327~381)
def load(self, weights, verbose=True):
    model = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights  # ema first
    csd = model.float().state_dict()  # checkpoint state_dict as FP32
    # Detect cls branch width (c3) depends on nc, so rebuild it at the source width to transfer its hidden layers,
    # but only when the head input widths match, since a branch from another model scale cannot run on this neck
    head, src_head = self.model[-1], getattr(model, "model", [None])[-1]
    if (
        isinstance(head, Detect)
        and isinstance(src_head, Detect)
        and not (self.is_fused() or model.is_fused())
        and [m[0].conv.in_channels for m in head.cv2] == [m[0].conv.in_channels for m in src_head.cv2]
    ):
        for attr in ("cv3", "one2one_cv3"):
            dst, src = getattr(head, attr, None), getattr(src_head, attr, None)
            if dst is not None and src is not None and dst[0][-1].in_channels != src[0][-1].in_channels:
                src = deepcopy(src)
                for s, d in zip(src, dst):
                    if s[-1].out_channels != d[-1].out_channels:  # new class count
                        s[-1] = nn.Conv2d(s[-1].in_channels, d[-1].out_channels, 1)
                    for sp, dp in zip(s.parameters(), d.parameters()):
                        sp.requires_grad_(dp.requires_grad)
                setattr(head, attr, src.to(dst[0][-1].weight.device, dst[0][-1].weight.dtype).train(dst.training))
                head.bias_init()
    # Remap classification head rows by class-name when nc differs (e.g. Obj365 -> COCO fine-tune)
    cls_remapped = self._remap_cls_by_names(csd, model, verbose=verbose)
    updated_csd = intersect_dicts(csd, self.state_dict())  # intersect
    self.load_state_dict(updated_csd, strict=False)  # load
    len_updated_csd = len(updated_csd) + cls_remapped
    first_conv = "model.0.conv.weight"  # hard-coded to yolo models, mostly used to boost multi-channel training
    w, src = self.state_dict().get(first_conv), csd.get(first_conv)
    if first_conv not in updated_csd and w is not None and src is not None and w.shape[2:] == src.shape[2:]:
        c1, c2 = min(len(w), len(src)), min(w.shape[1], src.shape[1])
        w[:c1, :c2] = src[:c1].sum(1, keepdim=True) if c2 == 1 else src[:c1, :c2]  # gray input sums RGB filters
        len_updated_csd += 1
    if len_updated_csd == len(self.state_dict()) and not hasattr(self, "set_classes"):  # World/YOLOE use embeddings
        names = getattr(model, "names", {})
        if getattr(self, "names", None) == {i: str(i) for i in range(len(names))}:  # YAML placeholder names
            self.names = dict(names)
    self.pt_path = getattr(model, "pt_path", None)  # provenance follows the weights selected above
    if verbose:
        LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")
        if getattr(model, "is_fused", lambda: False)() and not self.is_fused():
            LOGGER.warning("Pretrained weights are fused for inference; train from the unfused checkpoint instead.")

# ── utils/torch_utils.py  intersect_dicts  (줄 723~734)
def intersect_dicts(da, db, exclude=()):
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

# ── nn/tasks.py  DetectionModel.init_criterion  (줄 651~653)
def init_criterion(self):
    return E2ELoss(self) if getattr(self.model[-1], "one2one_cv2", None) is not None else v8DetectionLoss(self)
```

## 0-2. 데이터와 전처리

```python
# ── data/augment.py  v8_transforms  (줄 2835~2916)
def v8_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace) -> Compose:
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        size=(imgsz, imgsz),
        preserve_obb=getattr(dataset, "use_obb", False),
    )
    pre_transform = Compose([mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(1, CopyPaste(dataset, p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if getattr(dataset, "use_keypoints", False):
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            hyp.fliplr = hyp.flipud = 0.0  # both fliplr and flipud require flip_idx
            LOGGER.warning("No 'flip_idx' array defined in data.yaml, disabling 'fliplr' and 'flipud' augmentations.")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")
    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None), flip_idx=flip_idx),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms

# ── data/augment.py  Mosaic.apply_image  (줄 599~630)
def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
    layout = params["layout"]
    if self.n == 4:
        img4 = np.full((self.imgsz * 2, self.imgsz * 2, labels["img"].shape[2]), 114, dtype=np.uint8)
        for item in layout:
            labels_patch = item["labels_patch"]
            img = labels_patch["img"]
            x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
            x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        labels["img"] = img4
    elif self.n == 9:
        img9 = np.full((self.imgsz * 3, self.imgsz * 3, labels["img"].shape[2]), 114, dtype=np.uint8)
        for item in layout:
            labels_patch = item["labels_patch"]
            img = labels_patch["img"]
            x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            padw, padh = item["padw"], item["padh"]
            x1b, y1b = x1 - padw, y1 - padh
            x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
            img9[y1:y2, x1:x2] = img[y1b:y2b, x1b:x2b]
        labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
    return labels

# ── data/augment.py  LetterBox.get_params  (줄 1779~1826)
def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
    img = labels["img"]
    shape = img.shape[:2]  # current shape [height, width]
    new_shape = labels.pop("rect_shape", self.new_shape)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = max(round(shape[1] * r), 1), max(round(shape[0] * r), 1)  # keep thin images at least 1 pixel
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if self.auto:  # minimum rectangle
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
    elif self.scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
    if self.center:
        dw /= 2  # divide padding into 2 sides
        dh /= 2
    top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
    left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)
    return {
        "orig_shape": shape,
        "new_shape": new_shape,
        "ratio": ratio,
        "new_unpad": new_unpad,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }

# ── models/yolo/detect/train.py  DetectionTrainer.preprocess_batch  (줄 105~135)
def preprocess_batch(self, batch: dict) -> dict:
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
    batch["img"] = batch["img"].float() / 255
    if self.args.multi_scale > 0.0:
        imgs = batch["img"]
        sz = (
            random.randrange(
                max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),  # min imgsz
                int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),  # max imgsz
            )
            // self.stride
            * self.stride
        )  # size
        sf = sz / max(imgs.shape[2:])  # scale factor
        if sf != 1:
            ns = [
                math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
            ]  # new shape (stretched to gs-multiple)
            imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
        batch["img"] = imgs
    return batch
```

## 0-3. 학습 준비: batch 누적·옵티마이저·학습률

```python
# ── engine/trainer.py  BaseTrainer._build_train_pipeline  (줄 305~343)
def _build_train_pipeline(self):
    batch_size = self.batch_size // max(self.world_size, 1)
    self.train_loader = self.get_dataloader(
        self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
    )
    final_batch_size = len(self.train_loader.sampler) % self.train_loader.batch_size or self.train_loader.batch_size
    min_imgsz = max(self.stride, int(self.args.imgsz * (1 - self.args.multi_scale))) // self.stride * self.stride
    if min_imgsz < 2 * self.stride and not self.train_loader.drop_last and final_batch_size == 1:
        raise ValueError(
            f"final batch=1 training at imgsz={min_imgsz} gives BatchNorm a single value per channel; "
            f"change batch, or use imgsz and multi_scale that keep every size >= {2 * self.stride}"
        )
    # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
    self.test_loader = self.get_dataloader(
        self.data[self.args.split],
        batch_size=batch_size if self.args.task in {"obb", "semantic", "depth"} else batch_size * 2,
        rank=LOCAL_RANK,
        mode="val",
    )
    self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
    weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
    if not self.args.time:  # time mode re-estimates epochs after the first epoch
        self.accumulate = min(self.accumulate, len(self.train_loader) * self.epochs)
    if self.optimizer is None:
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()
    else:
        for group in self.optimizer.param_groups:
            if group.get("param_group") in {"weight", "muon"}:
                group["weight_decay"] = weight_decay

# ── engine/trainer.py  BaseTrainer.build_optimizer  (줄 1171~1262)
def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
    g = [{}, {}, {}, {}]  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSprop", "SGD", "MuSGD", "auto"}
    name = {x.lower(): x for x in optimizers}.get(str(name).lower(), str(name))
    if name == "auto":
        LOGGER.info(
            f"{colorstr('optimizer:')} 'optimizer=auto' found, "
            f"ignoring 'lr0={self.args.lr0}' and determining best 'optimizer' and 'lr0' automatically... "
        )
        nc = self.data.get("nc", 10)  # number of classes
        lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
        name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
        self.args.optimizer, self.args.lr0 = name, lr  # resume rebuilds this choice from train_args
        self.args.warmup_bias_lr = 0.0  # start bias warmup at zero
    use_muon = name == "MuSGD"
    for module_name, module in unwrap_model(model).named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if param.ndim in {2, 4} and use_muon:  # muon only orthogonalizes matrices and conv filters
                g[3][fullname] = param  # muon params
            elif "bias" in fullname:  # bias (no decay)
                g[2][fullname] = param
            elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                g[1][fullname] = param
            else:  # weight (with decay)
                g[0][fullname] = param
    if not use_muon:
        g = [x.values() for x in g[:3]]  # convert to list of params
    if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
        optim_args = {"lr": lr, "betas": (momentum, 0.999), "weight_decay": 0.0}
    elif name == "RMSprop":
        optim_args = {"lr": lr, "momentum": momentum}
    elif name == "SGD" or name == "MuSGD":
        optim_args = {"lr": lr, "momentum": momentum, "nesterov": True}
    else:
        raise NotImplementedError(
            f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
            "Request support for additional optimizers at https://github.com/ultralytics/ultralytics."
        )
    num_params = [len(g[0]), len(g[1]), len(g[2])]  # number of param groups
    g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
    g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
    g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
    muon, sgd = (0.2, 1.0)
    if use_muon:
        num_params[0] = len(g[3])  # update number of params
        g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
        # higher lr for certain parameters in MuSGD when finetuning
        target = unwrap_model(model)
        head = getattr(target, "student_model", target).model[-1]
        heads = (getattr(head, "cv3", None), getattr(head, "one2one_cv3", None))
        boosted = {id(p) for m in heads if m for p in m.parameters()}
        g_ = []  # new param groups
        for x in g:
            p = x.pop("params")
            p1, p2 = [], []
            for k, v in p.items():
                (p1 if id(v) in boosted or "proto.semseg" in k or "SemanticSegment" in k else p2).append(v)
            g_.extend([{"params": p1, **x, "lr": lr * 3}, {"params": p2, **x}])
        g = g_
    # fused=True must go to the constructor: that is where Adam registers _step_supports_amp_scaling, which lets
    # GradScaler pass found_inf to the kernel instead of reading it back on the host
    fused = {"fused": True} if name in {"Adam", "AdamW"} and TORCH_2_0 and next(model.parameters()).is_cuda else {}
    optimizer = (partial(MuSGD, muon=muon, sgd=sgd) if use_muon else getattr(optim, name))(params=g, **fused)
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
        f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
    )
    return optimizer

# ── engine/trainer.py  BaseTrainer._setup_scheduler  (줄 270~276)
def _setup_scheduler(self):
    if self.args.cos_lr:
        self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
    else:
        self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
    self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

# ── engine/trainer.py  BaseTrainer._get_warmup_iterations  (줄 278~281)
def _get_warmup_iterations(self, num_batches):
    warmup_epochs = min(self.args.warmup_epochs, max(self.epochs - 1, 0))
    return round(warmup_epochs * num_batches) if warmup_epochs > 0 else 0
```

## 0-4. 학습 반복 (전체)

```python
# ── engine/trainer.py  BaseTrainer._do_train  (줄 472~717)
def _do_train(self):
    if self.world_size > 1:
        self._setup_ddp()
    self._setup_train()
    nb = len(self.train_loader)  # number of batches
    nw = self._get_warmup_iterations(nb)
    last_opt_step = -1
    acc0 = self.accumulate  # nominal (already capped) accumulate; warmup below ramps up to it, never past it
    self.epoch_time = None
    self.epoch_time_start = time.time()
    self.train_time_start = time.time()
    self.run_callbacks("on_train_start")
    LOGGER.info(
        f"Using {len(self.train_loader.dataset)} train, {len(self.test_loader.dataset)} val images for "
        f"fraction={self.args.fraction} at imgsz={self.args.imgsz}\n"
        f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
        f"Logging results to {colorstr('bold', self.save_dir)}\n"
        f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
    )
    if self.args.close_mosaic:
        base_idx = (self.epochs - self.args.close_mosaic) * nb
        self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    epoch = self.start_epoch
    mosaic_closed = not self.args.close_mosaic  # close once when the run enters its final close_mosaic epochs
    self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
    self._oom_retries = 0  # OOM auto-reduce counter for first epoch
    while True:
        self.epoch = epoch
        self.run_callbacks("on_train_epoch_start")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
            self.scheduler.step()
        self._model_train()
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(self.train_loader)
        # Update dataloader attributes (optional)
        if not mosaic_closed and epoch >= self.epochs - self.args.close_mosaic:
            mosaic_closed = True
            self._close_dataloader_mosaic()
            self.train_loader.reset()
        if RANK in {-1, 0}:
            if self.loss_names:
                LOGGER.info(self.progress_string())
            pbar = TQDM(enumerate(self.train_loader), total=nb)
        self.tloss = None
        for i, batch in pbar:
            self.run_callbacks("on_train_batch_start")
            # Warmup
            ni = i + nb * epoch
            if nw and ni <= nw:  # ni == nw lands accumulate, lr and momentum exactly on their targets
                xi = [0, nw]  # x interp
                self.accumulate = max(1, int(np.interp(ni, xi, [1, acc0]).round()))
                for x in self.optimizer.param_groups:
                    # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = float(
                        np.interp(
                            ni,
                            xi,
                            [
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                    )
                    if "momentum" in x:
                        x["momentum"] = float(np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum]))
            do_step = ni - last_opt_step >= self.accumulate
            no_sync = self.world_size > 1 and not self.args.compile and not do_step  # all-reduce on steps only
            # Forward
            try:
                with self.model.no_sync() if no_sync else nullcontext():
                    with autocast(torch.bfloat16 if self.args.amp == "bf16" else self.amp, device=self.device.type):
                        batch = self.preprocess_batch(batch)
                        if self.args.compile:
                            # Decouple inference and loss calculations for improved compile performance
                            preds = self.model(batch["img"])
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        else:
                            loss, self.loss_items = self.model(batch)
                        self.loss = loss.sum()
                        if RANK != -1:
                            self.loss *= self.world_size
                        if not self.loss_names:  # derive loss names from the criterion's loss dict on first batch
                            self.loss_names = tuple(self.loss_items)
                            if RANK in {-1, 0}:
                                LOGGER.info(self.progress_string())
                                self.metrics.update(dict.fromkeys(self.label_loss_items(prefix="val"), 0.0))
                        self.tloss = (
                            self.loss_items
                            if self.tloss is None
                            else {k: (self.tloss[k] * i + v) / (i + 1) for k, v in self.loss_items.items()}
                        )
                    # Backward
                    self.scaler.scale(self.loss).backward()
            except RuntimeError as e:
                is_oom = "out of memory" in str(e).lower()  # torch.cuda.OutOfMemoryError requires torch>=1.13
                if not is_oom and not any(
                    s in str(e)
                    for s in (
                        "CUBLAS_STATUS_ALLOC_FAILED",
                        "CUDNN_STATUS_INTERNAL_ERROR",
                        "unable to find an engine",
                    )
                ):
                    raise
                if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                    raise  # only auto-reduce during first epoch on single GPU, max 3 retries
                self._oom_retries += 1
                old_batch = self.train_loader.batch_size  # the loader clamps batch to the dataset size
                self.args.batch = self.batch_size = max(old_batch // 2, 1)
                error = f"{self.device.type.upper()} out of memory" if is_oom else "CUDA backend memory error"
                LOGGER.warning(
                    f"{error} with batch={old_batch}. "
                    f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                )
                batch = loss = preds = None
                self.loss = self.loss_items = self.tloss = None
                if hasattr(self.train_loader, "close"):
                    self.train_loader.close()  # free the replaced loader's workers and prefetched batches
                self._clear_memory()
                self._build_train_pipeline()  # retain optimizer state across OOM retries
                mosaic_closed = not self.args.close_mosaic  # the rebuilt loader reopened mosaic, re-arm the gate
                self.validator.dataloader = self.test_loader  # the validator holds the pre-halving loader
                self.scheduler.last_epoch = self.start_epoch - 1
                nb = len(self.train_loader)
                nw = self._get_warmup_iterations(nb)
                last_opt_step = -1
                acc0 = self.accumulate  # the rebuild recomputed the nominal for the reduced batch size
                self.optimizer.zero_grad()
                break  # restart epoch loop with reduced batch size
            if do_step:
                self.optimizer_step()
                last_opt_step = ni
                # Timed stopping
                if self.args.time:
                    self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                    if RANK != -1:  # if DDP training
                        broadcast_list = [self.stop if RANK == 0 else None]
                        dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                        self.stop = broadcast_list[0]
                    if self.stop:  # training time exceeded
                        break
            # Log
            if RANK in {-1, 0}:
                loss_length = len(self.tloss)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                    % (
                        f"{epoch + 1}/{self.epochs}",
                        f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                        *self.tloss.values(),  # losses
                        batch.get("cls", batch["img"]).shape[0],  # no. of instances
                        batch["img"].shape[-1],  # imgsz, i.e 640
                    )
                )
                if self.args.plots and ni in self.plot_idx:
                    self.plot_training_samples(batch, ni)
            self.run_callbacks("on_train_batch_end")
            if self.stop:
                break  # allow external stop (e.g. platform cancellation) between batches
        else:
            # for/else: this block runs only when the for loop completes without break (no OOM retry)
            self._oom_retries = 0  # reset OOM counter after successful first epoch
        if self._oom_retries and not self.stop:
            continue  # OOM recovery broke the for loop, restart with reduced batch size
        if hasattr(unwrap_model(self.model).criterion, "update"):
            unwrap_model(self.model).criterion.update()
        self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers
        self.run_callbacks("on_train_epoch_end")
        if RANK in {-1, 0}:
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
        # Validation
        final_epoch = epoch + 1 >= self.epochs
        if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
            self._clear_memory(None if self.device.type == "mps" else 0.5)  # prevent VRAM spike
            self.metrics, self.fitness = self.validate()
        # NaN recovery
        if self._handle_nan_recovery(epoch):
            last_opt_step = -1  # redo the epoch with normal step cadence, like the OOM restart
            self.optimizer.zero_grad()  # drop the corrupted pass's gradients, including NaNs still in .grad
            continue
        self.nan_recovery_attempts = 0
        if RANK in {-1, 0}:
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            if self.args.time:
                self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
            # Save model
            if (self.args.save or final_epoch) and self.save_model():
                self.run_callbacks("on_model_save")
        # Scheduler
        t = time.time()
        self.epoch_time = t - self.epoch_time_start
        self.epoch_time_start = t
        if self.args.time:
            mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
            self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
            nw = self._get_warmup_iterations(nb)
            self._setup_scheduler()
            self.scheduler.last_epoch = self.epoch  # do not move
            self.stop |= epoch >= self.epochs  # stop if exceeded epochs
        self.run_callbacks("on_fit_epoch_end")
        # clear if memory utilization > 50%; always clear on MPS due to leak https://github.com/ultralytics/ultralytics/issues/22621
        self._clear_memory(None if self.device.type == "mps" else 0.5)
        # Early Stopping
        if RANK != -1:  # if DDP training
            broadcast_list = [self.stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            self.stop = broadcast_list[0]
        if self.stop:
            break  # must break all DDP ranks
        epoch += 1
    seconds = time.time() - self.train_time_start
    LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
    # Do final val with best.pt
    self.final_eval()
    if RANK in {-1, 0}:
        if self.args.plots:
            self.plot_metrics()
        self.run_callbacks("on_train_end")
    self._clear_memory()
    for loader in (self.train_loader, self.test_loader):
        if hasattr(loader, "close"):
            loader.close()  # shut down persistent dataloader workers so none survive to interpreter exit
```

## 0-5. 예측: Detect 층과 DFL

```python
# ── nn/modules/head.py  Detect.__init__  (줄 108~143)
def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: list[int] | tuple[int, ...] = ()):
    super().__init__()
    self.nc = nc  # number of classes
    self.nl = len(ch)  # number of detection layers
    self.reg_max = reg_max  # DFL channels
    self.no = nc + self.reg_max * 4  # number of outputs per anchor
    self.stride = torch.zeros(self.nl)  # strides computed during build
    c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
    self.cv2 = nn.ModuleList(
        nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
    )
    self.cv3 = (
        nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        if self.legacy
        else nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
    )
    self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    if end2end:
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)

# ── nn/modules/head.py  Detect.forward_head  (줄 169~178)
def forward_head(
    self, x: list[torch.Tensor], box_head: nn.Module | None = None, cls_head: nn.Module | None = None
) -> dict[str, torch.Tensor]:
    if box_head is None or cls_head is None:  # for fused inference
        return {}
    bs = x[0].shape[0]  # batch size
    boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
    scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
    return {"boxes": boxes, "scores": scores, "feats": x}

# ── nn/modules/head.py  Detect.forward  (줄 180~204)
def forward(
    self, x: list[torch.Tensor]
) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    preds = self.forward_head(x, **self.one2many)
    if getattr(self, "one2one_cv2", None) is not None:
        x_detach = [xi.detach() for xi in x] if self.training else x  # detach keeps one2one out of the backbone
        one2one = self.forward_head(x_detach, **self.one2one)
        preds = {"one2many": preds, "one2one": one2one}
    if self.training:
        return preds
    y = self._inference(preds["one2one" if self.end2end else "one2many"] if "one2one" in preds else preds)
    if self.end2end:
        y = self.postprocess(y.permute(0, 2, 1))
    return y if self.export else (y, preds)

# ── nn/modules/head.py  Detect.bias_init  (줄 229~241)
def bias_init(self):
    for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
        a[-1].bias.data[:] = 2.0  # box
        b[-1].bias.data[: self.nc] = math.log(
            5 / self.nc / (640 / self.stride[i]) ** 2
        )  # cls (.01 objects, 80 classes, 640 img)
    if getattr(self, "one2one_cv2", None) is not None:
        for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)

# ── nn/modules/block.py  DFL  (줄 60~81)
class DFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

# ── utils/tal.py  make_anchors  (줄 539~565)
def make_anchors(feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype = feats[0].dtype
    for i in range(len(feats)):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        # no cumsum (nondeterministic on CUDA), no device= (baked into traces), no out= (does not convert to CoreML)
        sx = torch.arange(w).type_as(feats[0]) + grid_cell_offset  # shift x
        sy = torch.arange(h).type_as(feats[0]) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(feats[0].new_full((h * w, 1), stride, dtype=dtype))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

# ── utils/tal.py  dist2bbox  (줄 568~587)
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

# ── utils/tal.py  bbox2dist  (줄 590~605)
def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    x1y1, x2y2 = bbox.chunk(2, -1)
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    if reg_max is not None:
        dist = dist.clamp_(0, reg_max - 0.01)  # dist (lt, rb)
    return dist
```

## 0-6. 손실 계산

```python
# ── utils/loss.py  v8DetectionLoss  (줄 355~512)
class v8DetectionLoss:
    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):  # model must be de-paralleled
        device = next(model.parameters()).device  # get model device
        h = model.args  # hyperparameters
        m = model.model[-1]  # Detect() module
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        self.loss_names = "box_loss", "cls_loss", "dfl_loss" if self.use_dfl else "l1_loss"
        # Class weights for handling imbalanced datasets
        self.class_weights = getattr(model, "class_weights", None)
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)
        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            stride=self.stride.tolist(),
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            batch_idx = targets[:, 0].long()  # image index
            _, counts = batch_idx.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            n = counts.max()
            if self.device.type == "mps":  # power-of-2 buckets bound MPS per-shape driver memory (#26742)
                n = 1 << (int(n) - 1).bit_length()
            out = torch.zeros(batch_size, n, ne - 1, device=self.device)
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            offsets = offsets.cumsum(0)
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            out[batch_idx, within_idx] = targets[:, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape  # batch, anchors, channels
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = pred_dist.view(b, a, c // 4, 4).transpose(2,3).softmax(3).matmul(self.proj.type(pred_dist.dtype))
            # pred_dist = (pred_dist.view(b, a, c // 4, 4).softmax(2) * self.proj.type(pred_dist.dtype).view(1, 1, -1, 1)).sum(2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any], **assign_kwargs
    ) -> tuple:
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        # Targets
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            **assign_kwargs,
        )
        target_scores_sum = target_scores.sum().clamp_(min=1)  # on the device: no host sync
        # Cls loss with optional class weighting
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum  # BCE
        # Bbox loss: zero on an empty foreground, and pred_distri stays in the graph either way
        loss[0], loss[2] = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            anchor_points,
            target_bboxes / stride_tensor,
            target_scores,
            target_scores_sum,
            fg_mask,
            imgsz,
            stride_tensor,
        )
        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # dfl gain
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )  # loss(box, cls, dfl)
    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        return preds[1] if isinstance(preds, tuple) else preds
    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.loss(self.parse_output(preds), batch)
    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = preds["boxes"].shape[0]
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        return loss * batch_size, loss_detach
```

## 0-7. 짝짓기

```python
# ── utils/tal.py  TaskAlignedAssigner  (줄 14~393)
class TaskAlignedAssigner(nn.Module):
    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        stride: list | None = None,
        eps: float = 1e-9,
        topk2: int | None = None,
    ):
        super().__init__()
        self.topk = topk
        self.topk2 = topk2 or topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.stride = stride if stride is not None else [8, 16, 32]
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        self.eps = eps
        self._oom_warned = False
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **extra):
        self.bs, _, self.num_classes = pd_scores.shape
        self.n_max_boxes = gt_bboxes.shape[1]
        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                torch.zeros_like(pd_scores[..., 0]),
            )
        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **extra)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
        # Recover outside the except block so e.__traceback__ releases the failed attempt's GPU intermediates.
        bs, n_max_boxes = self.bs, self.n_max_boxes
        if not self._oom_warned:
            LOGGER.warning(
                f"CUDA out of memory in TaskAlignedAssigner with batch_size={bs} and max_num_obj={n_max_boxes}; "
                "retrying assignment one image at a time on GPU. Model forward batch size is unchanged."
            )
            self._oom_warned = True
        last_gt_idx = (
            mask_gt.squeeze(-1)
            .bool()
            .mul(torch.arange(1, n_max_boxes + 1, device=mask_gt.device))
            .amax(1)
            .clamp_(min=1)
            .tolist()
        )
        self.bs = 1
        results = None
        try:
            for i, self.n_max_boxes in enumerate(last_gt_idx):
                result = self._forward(
                    pd_scores[i : i + 1],
                    pd_bboxes[i : i + 1],
                    anc_points,
                    gt_labels[i : i + 1, : self.n_max_boxes],
                    gt_bboxes[i : i + 1, : self.n_max_boxes],
                    mask_gt[i : i + 1, : self.n_max_boxes],
                    **{k: v[i : i + 1] for k, v in extra.items()},
                )
                if results is None:
                    results = tuple(x.new_empty((bs, *x.shape[1:])) for x in result)
                for output, x in zip(results, result):
                    output[i] = x[0]
        finally:
            self.bs, self.n_max_boxes = bs, n_max_boxes
        return results
    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **extra):
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, **extra
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )
        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        # Normalize
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        overlaps *= mask_pos
        pos_overlaps = overlaps.amax(dim=-1, keepdim=True)  # b, max_num_obj
        align_metric.mul_(pos_overlaps).div_(pos_align_metrics + self.eps)
        norm_align_metric = align_metric.amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, **extra):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt, **extra
        )
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk.mul_(mask_in_gts).mul_(mask_gt.bool())
        return mask_pos, align_metric, overlaps
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, **extra):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        shape = self.bs, self.n_max_boxes, na
        indices = mask_gt.nonzero(as_tuple=True)
        bbox_scores = pd_scores[indices[0], indices[2], gt_labels[indices[0], indices[1], 0].long()]
        overlap_values = self.pair_overlaps(indices, gt_bboxes, pd_bboxes, **extra)
        align_values = bbox_scores.pow(self.alpha) * overlap_values.pow(self.beta)
        overlaps = torch.zeros(shape, dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        align_metric = torch.zeros(shape, dtype=align_values.dtype, device=pd_scores.device)
        overlaps[indices] = overlap_values
        align_metric[indices] = align_values
        return align_metric, overlaps
    def pair_overlaps(self, indices, gt_bboxes, pd_bboxes):
        return self.iou_calculation(gt_bboxes[indices[:2]], pd_bboxes[indices[0], indices[2]])
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
    def select_topk_candidates(self, metrics, topk_mask=None):
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)
        # Count how many of the topk lists select each anchor; scatter_add_ accumulates duplicate indices in one pass
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
        # Filter invalid bboxes
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor
    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)
        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        # Assigned target scores
        target_labels.clamp_(0)
        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int8,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        target_scores = target_scores * (fg_mask[:, :, None] > 0)
        return target_labels, target_bboxes, target_scores
    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride_val  # floor tiny sides so the pool grows monotonically
        gt_bboxes_xywh[..., 2:] = torch.where(
            (wh_mask * mask_gt).bool(),
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            gt_bboxes_xywh[..., 2:],
        )
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)
        lt, rb = gt_bboxes.unsqueeze(2).chunk(2, 3)  # (b, n_boxes, 1, 2) left-top, right-bottom
        mask = xy_centers[:, 0] - lt[..., 0] > eps
        mask &= xy_centers[:, 1] - lt[..., 1] > eps
        mask &= rb[..., 0] - xy_centers[:, 0] > eps
        mask &= rb[..., 1] - xy_centers[:, 1] > eps
        return mask
    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        # Anchors assigned to multiple gt_bboxes keep the highest overlap; a no-op when there are none, without a sync
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
        max_overlaps_idx = overlaps.max(1).indices  # (b, h*w)
        is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (b, n_max_boxes, h*w)
        if self.topk2 != self.topk:
            align_metric = align_metric * mask_pos  # update overlaps
            # (b, n_max_boxes, topk2)
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)  # update mask_pos
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            mask_pos *= topk_idx
        # Column max gives fg flag + gt idx; NPU aclnnMaxDim lacks INT8 through CANN 9.1.1; cast 0/1 mask to int64
        fg_mask, target_gt_idx = (mask_pos.to(torch.int64) if mask_pos.device.type == "npu" else mask_pos).max(-2)
        return target_gt_idx, fg_mask, mask_pos
```

## 0-8. 박스 손실과 DFL 손실

```python
# ── utils/loss.py  BboxLoss  (줄 119~164)
class BboxLoss(nn.Module):
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fg_mask = fg_mask.nonzero(as_tuple=True)  # index once; long-index gathers and their backward do not sync
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # normalize ltrb by image size
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist = pred_dist * stride
            pred_dist[..., 0::2] /= imgsz[1]
            pred_dist[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl

# ── utils/loss.py  DFLoss  (줄 97~116)
class DFLoss(nn.Module):
    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()  # target left
        tr = tl + 1  # target right
        wl = tr - target  # weight left
        wr = 1 - wl  # weight right
        # Compute log_softmax once, then two gathers; cross_entropy(x, t) = -log_softmax(x).gather(t)
        logp = F.log_softmax(pred_dist, dim=1)
        return -(
            logp.gather(1, tl.view(-1, 1)).view(tl.shape) * wl + logp.gather(1, tr.view(-1, 1)).view(tl.shape) * wr
        ).mean(-1, keepdim=True)
```

## 0-9. 가중치 고치기와 EMA

```python
# ── engine/trainer.py  BaseTrainer.optimizer_step  (줄 925~936)
def optimizer_step(self):
    self.scaler.unscale_(self.optimizer)  # unscale gradients
    if self.device.type == "npu" and TORCH_2_0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0, foreach=False)
    else:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    self.optimizer.zero_grad()
    if self.ema:
        self.ema.update(self.model)

# ── utils/torch_utils.py  ModelEMA  (줄 817~892)
class ModelEMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = deepcopy(unwrap_model(model)).eval()  # FP32 EMA
        if hasattr(self.ema, "teacher_model"):
            # DistillationModel: strip the teacher so the EMA does not carry a full duplicate copy.
            self.ema.teacher_model = None
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True
        self._pairs = None  # (ema tensors, model tensors) with floating dtype, built on the first update
    def update(self, model):
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)
            if self._pairs is None:  # the tensors are updated in place, so the lists are built once
                msd = unwrap_model(model).state_dict()  # model state_dict
                ema_v, model_v = [], []
                for k, v in self.ema.state_dict().items():
                    if v.dtype.is_floating_point:  # true for FP16 and FP32
                        ema_v.append(v)
                        model_v.append(msd[k])
                self._pairs = ema_v, model_v
            ema_v, model_v = self._pairs
            if (
                ema_v and TORCH_2_0 and ema_v[0].device.type != "npu" and (TORCH_2_4 or ema_v[0].device.type != "mps")
            ):  # one kernel launch per op
                torch._foreach_lerp_(ema_v, model_v, 1 - d)
            else:  # _foreach_lerp_ needs torch>=2.0, MPS torch>=2.4, and is unavailable on NPU
                for v, m in zip(ema_v, model_v):
                    v.mul_(d).add_(m, alpha=1 - d)
    def update_attr(self, model, include=(), exclude=()):
        if self.enabled:
            copy_attr(self.ema, unwrap_model(model), include, exclude)
```

## 0-10. 검증: NMS와 best.pt 기준

```python
# ── utils/nms.py  non_max_suppression  (줄 13~157)
def non_max_suppression(
    prediction,
    conf_thres: float = 0.25,
    iou_thres: float = PREDICT_IOU,
    classes=None,
    agnostic: bool = False,
    multi_label: bool = False,
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    max_time_img: float = 0.05,
    max_nms: int = 30000,
    max_wh: int = 7680,
    rotated: bool = False,
    end2end: bool = False,
    return_idxs: bool = False,
):
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # model in validation mode, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)
    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output, keepi = [], []
        for pred in prediction:
            mask = pred[:, 4] > conf_thres
            if classes is not None:
                mask &= (pred[:, 5:6] == classes).any(1)
            idx = mask.nonzero(as_tuple=False).view(-1)[:max_det]
            output.append(pred[idx].float())  # half backend outputs: match the float32 rows the NMS branch returns
            keepi.append(idx)
        return (output, keepi) if return_idxs else output
    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    extra = prediction.shape[1] - nc - 4  # number of extra info
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]  # to track idxs
    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    t = time.time()
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    keepi = [torch.empty(0, dtype=torch.long, device=prediction.device)] * bs  # to store the kept idxs
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):  # image index, (preds, preds indices)
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        filt = xc[xi]  # confidence
        x = x[filt]
        if not rotated:
            x[:, :4] = xywh2xyxy(x[:, :4])
        if return_idxs:
            xk = xk[filt]
        # If none remain process next image
        if not x.shape[0]:
            continue
        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, extra), 1)
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            if return_idxs:
                xk = xk[filt]
        # Filter by class
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            filt = x[:, 4].argsort(descending=True)[:max_nms]  # sort by confidence and remove excess boxes
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1) if rotated else x[:, :4] + c
        i = nms_boxes(boxes, scores, iou_thres, rotated=rotated)
        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded
    return (output, keepi) if return_idxs else output

# ── utils/metrics.py  Metric.fitness  (줄 1013~1016)
def fitness(self) -> float:
    w = [0.0, 0.0, 0.0, 1.0]  # weights for [P, R, mAP50, mAP50-95]
    return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())
```
