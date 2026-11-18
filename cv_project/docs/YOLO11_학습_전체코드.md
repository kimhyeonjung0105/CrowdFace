# YOLO11 학습 전체 코드 (Ultralytics 8.4.208 원문, 한 줄씩 한국어 주석)

- 학습(`model.train`)이 실제로 지나가는 함수를 실행 순서대로, 줄임 없이 이어 붙였다. 코드 줄은 원문 그대로이고, 원문의 영어 주석과 설명문(docstring)은 지우고 **각 줄 바로 위에 한국어 주석**을 달았다.
- 각 함수 위 `# ──` 줄 = 파일 위치(`.venv/Lib/site-packages/ultralytics/` 기준)와 원본 줄 번호.
- 주석의 숫자는 우리 학습 설정 기준: 클래스 1개(head), imgsz 960, batch 16(P2는 8), epochs 30, optimizer auto → AdamW.
- `(해설 N절)` = claude.ai 문서 'YOLO11 학습 과정 해설' https://claude.ai/artifact/QupsSEAb49zHyrgRCFdjTx 의 절.
- 용어: 예측 칸(anchor) = 특징 지도의 한 칸, 한 칸이 박스 1개를 예측. stride = 그 지도에서 한 칸이 원본 몇 픽셀인지(8·16·32).
- Ultralytics는 AGPL-3.0. 공부·과제용 인용이다.

## 0-1. 모델 만들기

```python
# ── models/yolo/detect/train.py  DetectionTrainer.get_model  (줄 184~200)
# 학습 시작 때 한 번 불림. cfg = 모델 구조(yolo11n.yaml 또는 우리 head_p2.yaml), weights = COCO 사전학습 가중치
def get_model(self, cfg: str | dict | None = None, weights: torch.nn.Module | None = None, verbose: bool = True):
    # 빈 모델을 구조대로 만든다. nc = 데이터 yaml의 클래스 수(우리는 1 = head), ch = 입력 채널 3(RGB)
    model = self.set_model_names_for_load(
        # DetectionModel = backbone + neck + Detect 층. 이때 Detect의 클래스 출력이 80개가 아니라 1개로 만들어짐
        DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
    )
    # 사전학습 가중치가 있으면(yolo11n.pt)
    if weights:
        # 모양이 같은 층만 골라 복사 = 전이학습 (아래 load) (해설 1절)
        model.load(weights)
    # 가중치가 들어간 모델을 돌려줌
    return model

# ── nn/tasks.py  DetectionModel.__init__  (줄 523~561)
# 탐지 모델의 생성자. cfg 기본값은 yolo26n이지만 우리는 yolo11n.yaml을 넘김
def __init__(self, cfg="yolo26n.yaml", ch=3, nc=None, verbose=True):
    # 부모(nn.Module) 초기화
    super().__init__()
    # yaml을 읽어 층을 차례로 쌓음. nc가 yaml 값(80)과 다르면 nc=1로 덮어씀
    _initialize_yolo_model(self, cfg, ch, nc, verbose)
    # 맨 마지막 층 = Detect (탐지 머리)
    m = self.model[-1]
    # Detect 계열이면 stride(한 칸이 몇 픽셀인지)를 실제로 재 본다
    if isinstance(m, Detect):
        # 256×256 빈 사진으로 시험 실행
        s = 256
        # 메모리 절약용 제자리 연산 설정을 Detect에도 전달
        m.inplace = self.inplace
        # 시험 실행 함수: 모델을 통과시켜 Detect에 들어간 특징 지도(feats)만 꺼냄
        def _forward(x):
            # 모델 전체 실행
            output = self.forward(x)
            # YOLO26 같은 두 갈래 모델이면 one2many 쪽을 씀 (YOLO11은 해당 없음)
            if "one2many" in output:
                output = output["one2many"]
            # 특징 지도 목록 [P3, P4, P5] (P2 모델은 [P2, P3, P4, P5])
            return output["feats"]
        # BatchNorm 통계가 이 시험 실행으로 바뀌지 않게 평가 모드
        self.model.eval()
        # Detect만 학습 모드로 둬야 feats를 돌려줌
        m.training = True
        # stride = 256 ÷ 지도 높이 → 256/32=8, 256/16=16, 256/8=32 (P2면 4가 추가)
        m.stride = torch.tensor([s / x.shape[-2] for x in _forward(torch.zeros(1, ch, s, s))])
        # 모델 전체에도 stride 기록
        self.stride = m.stride
        # 다시 학습 모드로
        self.model.train()
        # Detect 마지막 층의 bias를 '처음엔 머리가 거의 없다'는 값으로 초기화 (아래 bias_init)
        m.bias_init()
    # Detect가 아닌 모델(RT-DETR 등)은 stride 32로 고정
    else:
        self.stride = torch.Tensor([32])
    # BatchNorm 설정값·활성화 함수 등 초기화
    initialize_weights(self)
    # verbose면 층 수·파라미터 수·GFLOPs 출력 (우리 P2: 2.67M, 10.5 GFLOPs)
    if verbose:
        self.info()
        LOGGER.info("")

# ── nn/tasks.py  BaseModel.load  (줄 327~381)
# 사전학습 가중치를 지금 모델에 옮겨 담음 = 전이학습의 핵심 (해설 1절)
def load(self, weights, verbose=True):
    # .pt 파일 안에 EMA 가중치가 있으면 그것을, 없으면 model을 씀
    model = (weights.get("ema") or weights["model"]) if isinstance(weights, dict) else weights
    # 가중치를 '이름 → 텐서' 사전(state_dict)으로, 32비트 실수로 꺼냄
    csd = model.float().state_dict()
    # 지금 모델의 Detect(head)와 가져온 모델의 Detect(src_head)
    head, src_head = self.model[-1], getattr(model, "model", [None])[-1]
    # 둘 다 Detect이고, 합쳐진(fused) 상태가 아니고, 박스 갈래(cv2) 입력 채널 수가 같으면 → 클래스 갈래 중간층도 살린다
    if (
        isinstance(head, Detect)
        and isinstance(src_head, Detect)
        and not (self.is_fused() or model.is_fused())
        and [m[0].conv.in_channels for m in head.cv2] == [m[0].conv.in_channels for m in src_head.cv2]
    ):
        # 클래스 갈래 cv3 (그리고 YOLO26의 one2one_cv3)에 대해
        for attr in ("cv3", "one2one_cv3"):
            # dst = 지금 모델의 갈래, src = COCO 모델의 갈래
            dst, src = getattr(head, attr, None), getattr(src_head, attr, None)
            # 중간 채널 수(c3)가 다르면: COCO는 c3=max(64,min(80,100))=80, 우리 nc=1은 c3=64 → 다름
            if dst is not None and src is not None and dst[0][-1].in_channels != src[0][-1].in_channels:
                # COCO 갈래를 통째로 복사해서
                src = deepcopy(src)
                # 층마다(P3·P4·P5) 
                for s, d in zip(src, dst):
                    # 마지막 1×1 층의 출력(클래스 수)이 다르면(80 vs 1)
                    if s[-1].out_channels != d[-1].out_channels:
                        # 마지막 층만 새로 만든다: 80채널 입력 → 1채널(head) 출력, 가중치는 무작위
                        s[-1] = nn.Conv2d(s[-1].in_channels, d[-1].out_channels, 1)
                    # 학습 여부(requires_grad) 설정은 원래 갈래를 따름
                    for sp, dp in zip(s.parameters(), d.parameters()):
                        sp.requires_grad_(dp.requires_grad)
                # 지금 모델의 cv3를 'COCO 중간층 + 새 마지막 층'으로 교체 (장치·자료형 맞춤)
                setattr(head, attr, src.to(dst[0][-1].weight.device, dst[0][-1].weight.dtype).train(dst.training))
                # 새 마지막 층 bias 다시 초기화
                head.bias_init()
    # 클래스 이름이 겹치면(예: Obj365→COCO) 그 줄만 옮겨 줌. 'head'는 COCO에 없어 0개
    cls_remapped = self._remap_cls_by_names(csd, model, verbose=verbose)
    # 이름과 모양이 둘 다 같은 텐서만 남김 (아래 intersect_dicts) → backbone·neck은 대부분 통과, 80클래스 층은 탈락
    updated_csd = intersect_dicts(csd, self.state_dict())
    # 남은 것만 모델에 넣음. strict=False = 빠진 텐서가 있어도 오류 없이 무작위 값으로 둠
    self.load_state_dict(updated_csd, strict=False)
    # 옮긴 개수 세기 (head_base 493/499, P2 297/593)
    len_updated_csd = len(updated_csd) + cls_remapped
    # 첫 합성곱 층 이름. 입력 채널 수가 다를 때(흑백 등) 특별 처리용
    first_conv = "model.0.conv.weight"
    # 지금 모델과 가져온 모델의 첫 층 가중치
    w, src = self.state_dict().get(first_conv), csd.get(first_conv)
    # 첫 층이 모양 차이로 탈락했지만 커널 크기는 같으면 (우리는 둘 다 RGB라 해당 없음)
    if first_conv not in updated_csd and w is not None and src is not None and w.shape[2:] == src.shape[2:]:
        # 겹치는 출력·입력 채널 수
        c1, c2 = min(len(w), len(src)), min(w.shape[1], src.shape[1])
        # 흑백 입력이면 RGB 세 필터를 더해서 넣고, 아니면 겹치는 만큼만 복사
        w[:c1, :c2] = src[:c1].sum(1, keepdim=True) if c2 == 1 else src[:c1, :c2]
        # 옮긴 개수 +1
        len_updated_csd += 1
    # 전부 옮겨졌다면(같은 데이터로 이어 학습) 클래스 이름도 가져옴. 우리는 일부만 옮겨져 해당 없음
    if len_updated_csd == len(self.state_dict()) and not hasattr(self, "set_classes"):
        # 가져온 모델의 클래스 이름
        names = getattr(model, "names", {})
        # 지금 이름이 yaml 기본값('0','1',…)이면
        if getattr(self, "names", None) == {i: str(i) for i in range(len(names))}:
            # 가져온 이름으로 바꿈
            self.names = dict(names)
    # 어떤 .pt에서 왔는지 기록
    self.pt_path = getattr(model, "pt_path", None)
    # 화면 출력: "Transferred 493/499 items from pretrained weights"
    if verbose:
        LOGGER.info(f"Transferred {len_updated_csd}/{len(self.model.state_dict())} items from pretrained weights")
        # 추론용으로 합쳐진 가중치를 넣었으면 경고
        if getattr(model, "is_fused", lambda: False)() and not self.is_fused():
            LOGGER.warning("Pretrained weights are fused for inference; train from the unfused checkpoint instead.")

# ── utils/torch_utils.py  intersect_dicts  (줄 723~734)
# 두 가중치 사전에서 '이름이 같고, 제외 단어가 없고, 모양이 같은' 것만 남김
def intersect_dicts(da, db, exclude=()):
    # 예: model.23.cv3.0.2.weight 는 COCO가 [80,…], 우리가 [1,…] → 모양이 달라 빠짐 = Detect 층 교체 (해설 1절)
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

# ── nn/tasks.py  DetectionModel.init_criterion  (줄 651~653)
# 손실 함수 고르기
def init_criterion(self):
    # YOLO26처럼 one2one 갈래가 있으면 E2ELoss, YOLO11은 v8DetectionLoss (0-6절)
    return E2ELoss(self) if getattr(self.model[-1], "one2one_cv2", None) is not None else v8DetectionLoss(self)
```

## 0-2. 데이터와 전처리

```python
# ── data/augment.py  v8_transforms  (줄 2835~2916)
# 학습용 사진 변형(증강) 순서를 정함. 사진 1장을 꺼낼 때마다 이 순서대로 지나감 (해설 2절)
def v8_transforms(dataset, imgsz: int, hyp: IterableSimpleNamespace) -> Compose:
    # 모자이크: 사진 4장을 2×2로 붙임 (p=1.0 = 항상, 마지막 10 epoch만 끔)
    mosaic = Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic)
    # 무작위 기하 변환(회전·이동·확대축소·기울이기·원근)
    affine = RandomPerspective(
        # 회전 각도 범위 (기본 0 = 회전 안 함)
        degrees=hyp.degrees,
        # 이동 범위 (기본 0.1 = 사진 크기의 ±10%)
        translate=hyp.translate,
        # 확대·축소 범위 (기본 0.5 = 0.5~1.5배)
        scale=hyp.scale,
        # 기울이기 (기본 0)
        shear=hyp.shear,
        # 원근 (기본 0)
        perspective=hyp.perspective,
        # 결과 크기 = 960×960 (모자이크 1920×1920을 여기서 잘라 줄임)
        size=(imgsz, imgsz),
        # 회전 박스(OBB)용 설정, 우리는 해당 없음
        preserve_obb=getattr(dataset, "use_obb", False),
    )
    # 앞부분 = 모자이크 → 기하 변환
    pre_transform = Compose([mosaic, affine])
    # copy_paste 방식이 flip이면(기본) 모자이크 바로 뒤에 끼움
    if hyp.copy_paste_mode == "flip":
        # 물체를 오려 붙이는 증강. 기본 확률 0이라 실제로는 아무것도 안 함 (분할 마스크가 필요)
        pre_transform.insert(1, CopyPaste(dataset, p=hyp.copy_paste, mode=hyp.copy_paste_mode))
    # mixup 방식이면 다른 사진에서 오려 와 맨 뒤에 붙임
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose([Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic), affine]),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    # 좌우 뒤집을 때 키포인트(왼눈↔오른눈) 짝 정보. 탐지에는 없음
    flip_idx = dataset.data.get("flip_idx", [])
    # 키포인트(자세) 모델일 때만 검사
    if getattr(dataset, "use_keypoints", False):
        # 키포인트 개수
        kpt_shape = dataset.data.get("kpt_shape", None)
        # 짝 정보 없이 뒤집으면 왼쪽·오른쪽이 섞이므로
        if len(flip_idx) == 0 and (hyp.fliplr > 0.0 or hyp.flipud > 0.0):
            # 뒤집기를 끄고
            hyp.fliplr = hyp.flipud = 0.0
            # 경고
            LOGGER.warning("No 'flip_idx' array defined in data.yaml, disabling 'fliplr' and 'flipud' augmentations.")
        # 짝 정보 길이가 키포인트 수와 다르면 오류
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")
    # 전체 순서를 묶어 돌려줌
    return Compose(
        [
            # ① 모자이크 + 기하 변환
            pre_transform,
            # ② 두 사진 반투명 겹치기 (기본 확률 0 = 안 함)
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            # ③ 다른 사진 조각 붙이기 (기본 확률 0 = 안 함)
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            # ④ 흐림·흑백·밝기 대비 등 약한 무작위 변형 (albumentations 설치 시, 각 1% 확률)
            Albumentations(p=1.0, transforms=getattr(hyp, "augmentations", None), flip_idx=flip_idx),
            # ⑤ 색상·채도·밝기 흔들기 (기본 ±0.015 / ±0.7 / ±0.4)
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            # ⑥ 상하 뒤집기 (기본 확률 0 = 안 함)
            RandomFlip(direction="vertical", p=hyp.flipud, flip_idx=flip_idx),
            # ⑦ 좌우 뒤집기 (기본 확률 0.5)
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )

# ── data/augment.py  Mosaic.apply_image  (줄 599~630)
# 미리 정한 배치(layout)대로 4장(또는 9장)을 큰 캔버스에 붙임
def apply_image(self, labels: dict[str, Any], params: dict[str, Any] | None = None) -> dict[str, Any]:
    # 각 사진을 캔버스 어디에 붙일지 미리 계산된 정보
    layout = params["layout"]
    # 4장 모자이크 (기본)
    if self.n == 4:
        # 1920×1920 캔버스를 회색 114로 채움 → 사진이 안 닿는 빈 곳은 회색 (해설 2절)
        img4 = np.full((self.imgsz * 2, self.imgsz * 2, labels["img"].shape[2]), 114, dtype=np.uint8)
        # 4장을 하나씩
        for item in layout:
            # 이 칸에 들어갈 사진과 라벨
            labels_patch = item["labels_patch"]
            # 사진 픽셀
            img = labels_patch["img"]
            # a = 캔버스에서 붙일 자리 (좌상단 x,y ~ 우하단 x,y)
            x1a, y1a, x2a, y2a = item["x1a"], item["y1a"], item["x2a"], item["y2a"]
            # b = 원본 사진에서 잘라 올 부분
            x1b, y1b, x2b, y2b = item["x1b"], item["y1b"], item["x2b"], item["y2b"]
            # 잘라서 붙이기
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
        # 결과 사진을 모자이크로 바꿈 (박스 좌표는 다른 함수에서 같은 만큼 옮김)
        labels["img"] = img4
    # 9장 모자이크 (우리 설정에선 안 씀)
    elif self.n == 9:
        # 3배 크기 회색 캔버스
        img9 = np.full((self.imgsz * 3, self.imgsz * 3, labels["img"].shape[2]), 114, dtype=np.uint8)
        # 9장을 하나씩
        for item in layout:
            # 이 칸의 사진과 라벨
            labels_patch = item["labels_patch"]
            # 사진 픽셀
            img = labels_patch["img"]
            # 캔버스에서 붙일 자리
            x1, y1, x2, y2 = item["x1"], item["y1"], item["x2"], item["y2"]
            # 사진이 자리보다 바깥으로 튀어나온 만큼
            padw, padh = item["padw"], item["padh"]
            # 원본에서 잘라 올 시작점
            x1b, y1b = x1 - padw, y1 - padh
            # 원본에서 잘라 올 끝점 (자리 크기만큼)
            x2b, y2b = x1b + (x2 - x1), y1b + (y2 - y1)
            # 잘라서 붙이기
            img9[y1:y2, x1:x2] = img[y1b:y2b, x1b:x2b]
        # 가장자리(border)를 잘라 낸 가운데 부분만 씀
        labels["img"] = img9[-self.border[0] : self.border[0], -self.border[1] : self.border[1]]
    # 모자이크 사진을 담은 라벨 사전
    return labels

# ── data/augment.py  LetterBox.get_params  (줄 1779~1826)
# 비율을 유지한 채 줄이고, 남는 곳을 회색 띠로 채울 크기를 계산 (검증·모자이크 끈 뒤에 씀) (해설 2절)
def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
    # 원본 사진
    img = labels["img"]
    # 원본 크기 [높이, 너비]
    shape = img.shape[:2]
    # 목표 크기 (검증 때 rect 모드면 사진마다 다른 직사각형, 아니면 960)
    new_shape = labels.pop("rect_shape", self.new_shape)
    # 숫자 하나면
    if isinstance(new_shape, int):
        # 정사각형 (960, 960)
        new_shape = (new_shape, new_shape)
    # 줄이는 배율 = 높이·너비 중 더 많이 줄여야 하는 쪽 → 비율 유지
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # 작은 사진은 키우지 않는 설정이면
    if not self.scaleup:
        # 배율 1 이하로 제한
        r = min(r, 1.0)
    # 가로·세로 배율 (같음)
    ratio = r, r
    # 줄인 뒤 크기 (너비, 높이), 최소 1픽셀
    new_unpad = max(round(shape[1] * r), 1), max(round(shape[0] * r), 1)
    # 채워야 할 여백 = 목표 − 줄인 크기
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    # auto면 여백을 stride(32)의 배수가 되는 만큼만 최소로
    if self.auto:
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
    # scale_fill이면 비율 무시하고 늘려서 꽉 채움 (여백 0)
    elif self.scale_fill:
        dw, dh = 0.0, 0.0
        # 줄인 크기 = 목표 크기
        new_unpad = (new_shape[1], new_shape[0])
        # 가로·세로 배율이 달라짐
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    # 가운데 정렬이면
    if self.center:
        # 여백을 양쪽에 반씩
        dw /= 2
        dh /= 2
    # 위·아래 여백 픽셀 (0.1을 빼고 더해 홀수일 때 한쪽이 1px 더 갖게)
    top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
    # 왼쪽·오른쪽 여백 픽셀
    left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)
    # 계산값을 돌려줌 → 다음 단계가 사진을 줄이고 회색(114) 띠를 붙임
    return {
        # 원본 크기
        "orig_shape": shape,
        # 목표 크기
        "new_shape": new_shape,
        # 배율 (박스 좌표도 같이 곱함)
        "ratio": ratio,
        # 줄인 크기
        "new_unpad": new_unpad,
        # 위 여백
        "top": top,
        # 아래 여백
        "bottom": bottom,
        # 왼쪽 여백
        "left": left,
        # 오른쪽 여백
        "right": right,
    }

# ── models/yolo/detect/train.py  DetectionTrainer.preprocess_batch  (줄 105~135)
# 묶음(batch) 하나를 GPU로 보내고 픽셀 값을 0~1로 바꿈
def preprocess_batch(self, batch: dict) -> dict:
    # 묶음 안의 모든 항목(사진·박스·클래스…)에 대해
    for k, v in batch.items():
        # 텐서이면
        if isinstance(v, torch.Tensor):
            # GPU로 보냄 (CPU·Mac이 아니면 기다리지 않고 비동기로)
            batch[k] = v.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
    # 0~255 정수 → 0~1 실수. 정규화는 이것뿐 (평균·표준편차 빼기 없음) (해설 2절)
    batch["img"] = batch["img"].float() / 255
    # 다중 크기 학습(기본 0 = 끔)이면
    if self.args.multi_scale > 0.0:
        # 사진 묶음
        imgs = batch["img"]
        # 이번 묶음 크기를 무작위로 고름
        sz = (
            random.randrange(
                # 최소 = imgsz × (1 − 비율), stride 이상
                max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),
                # 최대 = imgsz × (1 + 비율)
                int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),
            )
            # stride(32)의 배수로 내림
            // self.stride
            * self.stride
        )
        # 바꿀 배율 = 새 크기 ÷ 지금 긴 변
        sf = sz / max(imgs.shape[2:])
        # 배율이 1이 아니면
        if sf != 1:
            # 새 높이·너비 (stride 배수로 올림)
            ns = [
                math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
            ]
            # 쌍선형 보간으로 크기 변경 (박스는 0~1 비율 좌표라 그대로)
            imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
        # 바뀐 사진을 다시 넣음
        batch["img"] = imgs
    # 준비된 묶음
    return batch
```

## 0-3. 학습 준비: batch 누적·옵티마이저·학습률

```python
# ── engine/trainer.py  BaseTrainer._build_train_pipeline  (줄 305~343)
# 데이터 로더·batch 누적 횟수·옵티마이저·학습률 계획을 한 번에 준비 (해설 3절)
def _build_train_pipeline(self):
    # GPU 여러 개면 나눠 가짐. 우리는 GPU 1개 → 16 그대로
    batch_size = self.batch_size // max(self.world_size, 1)
    # 학습 데이터 로더 (train 15,000장, 위 v8_transforms 증강 포함)
    self.train_loader = self.get_dataloader(
        self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
    )
    # 마지막 묶음 크기 (15,000 ÷ 16 나머지 8 → 8장)
    final_batch_size = len(self.train_loader.sampler) % self.train_loader.batch_size or self.train_loader.batch_size
    # 다중 크기 학습 때 가장 작은 입력 크기
    min_imgsz = max(self.stride, int(self.args.imgsz * (1 - self.args.multi_scale))) // self.stride * self.stride
    # 아주 작은 입력에 마지막 묶음이 1장이면 BatchNorm이 계산 불가 → 미리 오류
    if min_imgsz < 2 * self.stride and not self.train_loader.drop_last and final_batch_size == 1:
        raise ValueError(
            f"final batch=1 training at imgsz={min_imgsz} gives BatchNorm a single value per channel; "
            f"change batch, or use imgsz and multi_scale that keep every size >= {2 * self.stride}"
        )
    # 검증 데이터 로더 (val 4,370장, 증강 없음, batch 2배 = 32)
    self.test_loader = self.get_dataloader(
        # 검증용 split (기본 val)
        self.data[self.args.split],
        # 탐지는 batch × 2
        batch_size=batch_size if self.args.task in {"obb", "semantic", "depth"} else batch_size * 2,
        rank=LOCAL_RANK,
        mode="val",
    )
    # 몇 묶음마다 가중치를 고칠지 = 64 ÷ 16 = 4 (P2 batch 8이면 8) → 유효 batch 64로 같음 (해설 3절)
    self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
    # 가중치 감쇠도 유효 batch에 맞춰 조정: 0.0005 × 16 × 4 ÷ 64 = 0.0005
    weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
    # 시간 제한 학습이 아니면
    if not self.args.time:
        # 누적 횟수가 전체 묶음 수보다 크지 않게
        self.accumulate = min(self.accumulate, len(self.train_loader) * self.epochs)
    # 옵티마이저가 아직 없으면(처음 시작)
    if self.optimizer is None:
        # 전체 가중치 수정 횟수 = ⌈15,000 ÷ 64⌉ × 30 = 235 × 30 = 7,050 → auto 판단에 씀
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        # 옵티마이저 만들기 (아래 build_optimizer)
        self.optimizer = self.build_optimizer(
            # 학습할 모델
            model=self.model,
            # 기본 'auto'
            name=self.args.optimizer,
            # 기본 0.01 (auto면 무시됨)
            lr=self.args.lr0,
            # 기본 0.937
            momentum=self.args.momentum,
            # 위에서 계산한 가중치 감쇠
            decay=weight_decay,
            # 7,050
            iterations=iterations,
        )
        # 학습률 계획(epoch마다 얼마나 줄일지) 설정
        self._setup_scheduler()
    # 이미 있으면(메모리 부족으로 batch를 줄여 다시 만드는 경우)
    else:
        # 가중치 감쇠만 새 batch에 맞게 고침
        for group in self.optimizer.param_groups:
            if group.get("param_group") in {"weight", "muon"}:
                group["weight_decay"] = weight_decay

# ── engine/trainer.py  BaseTrainer.build_optimizer  (줄 1171~1262)
# 파라미터를 무리로 나누고 옵티마이저를 만듦 (해설 3절)
def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
    # 파라미터 무리 4개: [0] 감쇠 있는 가중치, [1] 감쇠 없는 가중치(BN), [2] bias, [3] MuSGD용
    g = [{}, {}, {}, {}]
    # 'Norm'이 이름에 든 층 종류 목록 (BatchNorm2d 등)
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)
    # 쓸 수 있는 옵티마이저 이름
    optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSprop", "SGD", "MuSGD", "auto"}
    # 대소문자 상관없이 이름 맞추기 ('adamw' → 'AdamW')
    name = {x.lower(): x for x in optimizers}.get(str(name).lower(), str(name))
    # 기본값 auto면 자동으로 고름
    if name == "auto":
        # 안내 출력: lr0 설정을 무시하고 자동으로 정한다
        LOGGER.info(
            f"{colorstr('optimizer:')} 'optimizer=auto' found, "
            f"ignoring 'lr0={self.args.lr0}' and determining best 'optimizer' and 'lr0' automatically... "
        )
        # 클래스 수 (우리 1)
        nc = self.data.get("nc", 10)
        # 경험식: 0.002 × 5 ÷ (4 + 1) = 0.002
        lr_fit = round(0.002 * 5 / (4 + nc), 6)
        # 수정 횟수가 1만 번 넘으면 MuSGD, 아니면 AdamW → 우리 7,050번이라 AdamW lr 0.002
        name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
        # 고른 값을 설정에 기록 (resume 때 같은 선택)
        self.args.optimizer, self.args.lr0 = name, lr
        # bias 워밍업 시작 학습률을 0으로
        self.args.warmup_bias_lr = 0.0
    # MuSGD를 쓰는지 (우리는 False)
    use_muon = name == "MuSGD"
    # 모델의 모든 층을 돌며
    for module_name, module in unwrap_model(model).named_modules():
        # 그 층이 직접 가진 파라미터마다
        for param_name, param in module.named_parameters(recurse=False):
            # 전체 이름 (예: model.0.conv.weight)
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            # MuSGD면 행렬·합성곱 필터는 [3]무리
            if param.ndim in {2, 4} and use_muon:
                g[3][fullname] = param
            # 이름에 bias가 있으면 [2]무리 (감쇠 없음)
            elif "bias" in fullname:
                g[2][fullname] = param
            # BatchNorm 가중치는 [1]무리 (감쇠 없음)
            elif isinstance(module, bn) or "logit_scale" in fullname:
                g[1][fullname] = param
            # 나머지 합성곱 가중치는 [0]무리 (감쇠 있음)
            else:
                g[0][fullname] = param
    # MuSGD가 아니면
    if not use_muon:
        # 이름은 버리고 파라미터만 목록으로 (앞 3무리)
        g = [x.values() for x in g[:3]]
    # Adam 계열이면 (우리 AdamW)
    if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
        # betas=(0.9, 0.999), 감쇠는 무리별로 따로 넣으므로 여기선 0
        optim_args = {"lr": lr, "betas": (momentum, 0.999), "weight_decay": 0.0}
    # RMSprop이면
    elif name == "RMSprop":
        optim_args = {"lr": lr, "momentum": momentum}
    # SGD 계열이면 네스테로프 모멘텀
    elif name == "SGD" or name == "MuSGD":
        optim_args = {"lr": lr, "momentum": momentum, "nesterov": True}
    # 모르는 이름이면 오류
    else:
        raise NotImplementedError(
            f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
            "Request support for additional optimizers at https://github.com/ultralytics/ultralytics."
        )
    # 무리별 개수 (출력용)
    num_params = [len(g[0]), len(g[1]), len(g[2])]
    # bias 무리: 감쇠 0
    g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
    # 합성곱 가중치 무리: 감쇠 0.0005 (과적합 방지)
    g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
    # BatchNorm 무리: 감쇠 0
    g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
    # MuSGD 섞는 비율 (우리는 안 씀)
    muon, sgd = (0.2, 1.0)
    # MuSGD일 때만 (아래 블록 전체 우리는 해당 없음)
    if use_muon:
        # 개수 갱신
        num_params[0] = len(g[3])
        # [3]무리 설정
        g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
        # 모델 꺼내기
        target = unwrap_model(model)
        # Detect 층
        head = getattr(target, "student_model", target).model[-1]
        # 클래스 갈래들
        heads = (getattr(head, "cv3", None), getattr(head, "one2one_cv3", None))
        # 클래스 갈래 파라미터 id 모음 → 이것만 학습률 3배
        boosted = {id(p) for m in heads if m for p in m.parameters()}
        # 새 무리 목록
        g_ = []
        # 무리마다
        for x in g:
            # 파라미터 꺼내기
            p = x.pop("params")
            # p1 = 학습률 3배, p2 = 그대로
            p1, p2 = [], []
            # 파라미터마다 나누기
            for k, v in p.items():
                (p1 if id(v) in boosted or "proto.semseg" in k or "SemanticSegment" in k else p2).append(v)
            # 두 무리로 쪼개 추가
            g_.extend([{"params": p1, **x, "lr": lr * 3}, {"params": p2, **x}])
        # 교체
        g = g_
    # GPU에서 Adam/AdamW면 여러 연산을 한 번에 하는 fused 모드 (빠름)
    fused = {"fused": True} if name in {"Adam", "AdamW"} and TORCH_2_0 and next(model.parameters()).is_cuda else {}
    # 옵티마이저 생성 = torch.optim.AdamW(무리 3개)
    optimizer = (partial(MuSGD, muon=muon, sgd=sgd) if use_muon else getattr(optim, name))(params=g, **fused)
    # 출력: "AdamW(lr=0.002, momentum=0.9) with parameter groups …"
    LOGGER.info(
        f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
        f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), {num_params[2]} bias(decay=0.0)"
    )
    # 만든 옵티마이저
    return optimizer

# ── engine/trainer.py  BaseTrainer._setup_scheduler  (줄 270~276)
# epoch마다 학습률에 곱할 배율(lf)을 정함 (해설 3절)
def _setup_scheduler(self):
    # cos_lr=True면 코사인 곡선으로 줄임 (기본 False)
    if self.args.cos_lr:
        self.lf = one_cycle(1, self.args.lrf, self.epochs)
    # 기본: 직선으로 1 → lrf(0.01)까지 → lr 0.002가 마지막엔 0.00002
    else:
        self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf
    # PyTorch 스케줄러에 연결: 매 epoch 시작에 lr = 처음 lr × lf(epoch)
    self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

# ── engine/trainer.py  BaseTrainer._get_warmup_iterations  (줄 278~281)
# 워밍업(처음에 학습률을 0에서 천천히 올리는 구간)이 몇 묶음인지
def _get_warmup_iterations(self, num_batches):
    # 워밍업 epoch = 3 (전체 epoch−1을 넘지 않게)
    warmup_epochs = min(self.args.warmup_epochs, max(self.epochs - 1, 0))
    # 3 × 938묶음 = 2,814묶음 동안 워밍업
    return round(warmup_epochs * num_batches) if warmup_epochs > 0 else 0
```

## 0-4. 학습 반복 (전체)

```python
# ── engine/trainer.py  BaseTrainer._do_train  (줄 472~717)
# 학습 전체 반복. epoch(전체 데이터 한 바퀴) × 묶음(batch) 이중 반복 (해설 3·6절)
def _do_train(self):
    # GPU 여러 개면 분산 학습 준비 (우리는 1개라 건너뜀)
    if self.world_size > 1:
        self._setup_ddp()
    # 모델·데이터·옵티마이저·EMA·손실 등 준비 (위 _build_train_pipeline 포함)
    self._setup_train()
    # 한 epoch의 묶음 수 = 938
    nb = len(self.train_loader)
    # 워밍업 묶음 수 = 2,814
    nw = self._get_warmup_iterations(nb)
    # 마지막으로 가중치를 고친 묶음 번호 (아직 없음)
    last_opt_step = -1
    # 목표 누적 횟수 4를 기억 (워밍업 동안 1 → 4로 올림)
    acc0 = self.accumulate
    # 한 epoch 걸린 시간 (아직 없음)
    self.epoch_time = None
    # epoch 시작 시각
    self.epoch_time_start = time.time()
    # 학습 전체 시작 시각
    self.train_time_start = time.time()
    # '학습 시작' 알림 (로그·외부 도구 연결용)
    self.run_callbacks("on_train_start")
    # 출력: 학습 15,000장 · 검증 4,370장, imgsz 960, 결과 저장 폴더, 30 epoch
    LOGGER.info(
        f"Using {len(self.train_loader.dataset)} train, {len(self.test_loader.dataset)} val images for "
        f"fraction={self.args.fraction} at imgsz={self.args.imgsz}\n"
        f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
        f"Logging results to {colorstr('bold', self.save_dir)}\n"
        f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
    )
    # 마지막 10 epoch에 모자이크를 끈다면
    if self.args.close_mosaic:
        # 모자이크 끈 직후 첫 묶음 번호
        base_idx = (self.epochs - self.args.close_mosaic) * nb
        # 그 묶음 3개를 그림으로 저장 (train_batch*.jpg)
        self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
    # 시작 epoch (처음이면 0, resume이면 이어서. 우리 P2는 22에서 resume)
    epoch = self.start_epoch
    # 모자이크를 이미 껐는지 (close_mosaic=10이면 아직 False)
    mosaic_closed = not self.args.close_mosaic
    # 남아 있을지 모르는 기울기를 0으로
    self.optimizer.zero_grad()
    # 메모리 부족 재시도 횟수
    self._oom_retries = 0
    # epoch 반복 (끝 조건은 안에서 break)
    while True:
        # 지금 epoch 기록
        self.epoch = epoch
        # 'epoch 시작' 알림
        self.run_callbacks("on_train_epoch_start")
        # 경고 메시지 숨기기
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 학습률 계획 한 칸 진행 → 이번 epoch lr = 0.002 × lf(epoch)
            self.scheduler.step()
        # 모델을 학습 모드로 (BatchNorm·Dropout이 학습용으로 동작)
        self._model_train()
        # 분산 학습이면 epoch마다 섞는 순서 바꿈
        if RANK != -1:
            self.train_loader.sampler.set_epoch(epoch)
        # 묶음 반복자 (번호, 묶음)
        pbar = enumerate(self.train_loader)
        # 마지막 10 epoch(21번째 epoch부터)에 들어서면
        if not mosaic_closed and epoch >= self.epochs - self.args.close_mosaic:
            # 한 번만
            mosaic_closed = True
            # 모자이크 끄기 → 실제 사진 모양(LetterBox)으로 마무리 학습 (해설 2절)
            self._close_dataloader_mosaic()
            # 로더 다시 시작
            self.train_loader.reset()
        # 메인 프로세스면
        if RANK in {-1, 0}:
            # 손실 이름이 정해졌으면 표 머리 출력 (Epoch GPU_mem box_loss cls_loss dfl_loss …)
            if self.loss_names:
                LOGGER.info(self.progress_string())
            # 진행 막대 표시
            pbar = TQDM(enumerate(self.train_loader), total=nb)
        # 이번 epoch 평균 손실 (초기화)
        self.tloss = None
        # 묶음 반복: i = 이번 epoch 안의 번호, batch = 사진 16장 + 라벨
        for i, batch in pbar:
            # '묶음 시작' 알림
            self.run_callbacks("on_train_batch_start")
            # 학습 시작부터 센 전체 묶음 번호
            ni = i + nb * epoch
            # 워밍업 구간(처음 2,814묶음)이면
            if nw and ni <= nw:
                # 보간 구간 [0, 2814]
                xi = [0, nw]
                # 누적 횟수를 1 → 4로 서서히 (처음엔 매 묶음 고침)
                self.accumulate = max(1, int(np.interp(ni, xi, [1, acc0]).round()))
                # 파라미터 무리마다
                for x in self.optimizer.param_groups:
                    # 학습률을 직선으로 올림
                    x["lr"] = float(
                        np.interp(
                            # 지금 위치
                            ni,
                            # 구간
                            xi,
                            [
                                # 시작값: bias는 warmup_bias_lr(auto라 0), 나머지 0
                                self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                # 끝값: 이번 epoch의 계획 lr
                                x["initial_lr"] * self.lf(epoch),
                            ],
                        )
                    )
                    # 'momentum' 칸이 있으면 (SGD만. 우리 AdamW는 betas를 써서 이 칸이 없어 건너뜀)
                    if "momentum" in x:
                        # 0.8 → 0.937로 서서히
                        x["momentum"] = float(np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum]))
            # 이번 묶음 뒤에 가중치를 고칠 차례인지 (마지막 수정 후 accumulate 묶음이 지났나)
            do_step = ni - last_opt_step >= self.accumulate
            # 분산 학습에서 고칠 차례가 아니면 GPU 간 동기화 생략
            no_sync = self.world_size > 1 and not self.args.compile and not do_step
            # 앞으로 계산(forward) 시작. 메모리 부족 오류를 잡기 위해 try
            try:
                # 동기화 생략 여부 적용
                with self.model.no_sync() if no_sync else nullcontext():
                    # 자동 혼합 정밀도(AMP): 일부 연산을 16비트로 → 빠르고 메모리 절약
                    with autocast(torch.bfloat16 if self.args.amp == "bf16" else self.amp, device=self.device.type):
                        # GPU로 보내고 /255 (0-2절 preprocess_batch)
                        batch = self.preprocess_batch(batch)
                        # torch.compile을 쓰면 (우리는 안 씀)
                        if self.args.compile:
                            # 예측만 먼저
                            preds = self.model(batch["img"])
                            # 손실은 따로
                            loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                        # 기본: 모델에 묶음을 통째로 넣으면 예측 → 손실까지 계산해 돌려줌 (0-5~0-8절)
                        else:
                            loss, self.loss_items = self.model(batch)
                        # 손실 3개(box·cls·dfl)를 더해 하나로
                        self.loss = loss.sum()
                        # 분산 학습이면 GPU 수만큼 곱함
                        if RANK != -1:
                            self.loss *= self.world_size
                        # 첫 묶음에서 손실 이름 정하기
                        if not self.loss_names:
                            # ('box_loss', 'cls_loss', 'dfl_loss')
                            self.loss_names = tuple(self.loss_items)
                            # 메인 프로세스면
                            if RANK in {-1, 0}:
                                # 표 머리 출력
                                LOGGER.info(self.progress_string())
                                # 검증 손실 칸 0으로 준비
                                self.metrics.update(dict.fromkeys(self.label_loss_items(prefix="val"), 0.0))
                        # 이번 epoch 평균 손실 갱신 (화면·results.csv용)
                        self.tloss = (
                            # 첫 묶음이면 그대로
                            self.loss_items
                            if self.tloss is None
                            # 아니면 누적 평균 = (이전 평균 × i + 이번 값) ÷ (i+1)
                            else {k: (self.tloss[k] * i + v) / (i + 1) for k, v in self.loss_items.items()}
                        )
                    # 뒤로 계산(backward): 손실을 각 가중치로 미분 → 기울기가 .grad에 '더해짐' (누적)
                    # scaler = 16비트에서 기울기가 0으로 사라지지 않게 손실을 크게 곱해 둠
                    self.scaler.scale(self.loss).backward()
            # 오류가 나면
            except RuntimeError as e:
                # GPU 메모리 부족인지
                is_oom = "out of memory" in str(e).lower()
                # 메모리 부족도 아니고, 메모리 관련 CUDA 오류도 아니면
                if not is_oom and not any(
                    s in str(e)
                    for s in (
                        "CUBLAS_STATUS_ALLOC_FAILED",
                        "CUDNN_STATUS_INTERNAL_ERROR",
                        "unable to find an engine",
                    )
                ):
                    # 그대로 오류를 냄
                    raise
                # 첫 epoch이 아니거나 3번 넘게 재시도했거나 분산 학습이면 포기
                if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                    raise
                # 재시도 횟수 +1
                self._oom_retries += 1
                # 지금 batch 크기
                old_batch = self.train_loader.batch_size
                # batch를 절반으로 (16 → 8)
                self.args.batch = self.batch_size = max(old_batch // 2, 1)
                # 경고 문구
                error = f"{self.device.type.upper()} out of memory" if is_oom else "CUDA backend memory error"
                # 출력: "batch=16에서 메모리 부족, 8로 줄여 다시"
                LOGGER.warning(
                    f"{error} with batch={old_batch}. "
                    f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                )
                # 메모리 비우기 위해 참조 끊기
                batch = loss = preds = None
                self.loss = self.loss_items = self.tloss = None
                # 기존 로더가 닫을 수 있으면
                if hasattr(self.train_loader, "close"):
                    # 작업 프로세스와 미리 읽은 묶음 해제
                    self.train_loader.close()
                # GPU 메모리 정리
                self._clear_memory()
                # 줄인 batch로 로더·누적 횟수 다시 만듦 (옵티마이저 상태는 유지)
                self._build_train_pipeline()
                # 새 로더는 모자이크가 켜져 있으니 다시 끌 준비
                mosaic_closed = not self.args.close_mosaic
                # 검증기에도 새 검증 로더 연결
                self.validator.dataloader = self.test_loader
                # 학습률 계획을 시작 epoch 직전으로 되돌림
                self.scheduler.last_epoch = self.start_epoch - 1
                # 새 묶음 수
                nb = len(self.train_loader)
                # 새 워밍업 길이
                nw = self._get_warmup_iterations(nb)
                # 마지막 수정 번호 초기화
                last_opt_step = -1
                # 새 목표 누적 횟수 (batch 8이면 8)
                acc0 = self.accumulate
                # 기울기 0으로
                self.optimizer.zero_grad()
                # 묶음 반복을 빠져나가 epoch을 처음부터 다시
                break
            # 가중치를 고칠 차례면 (4묶음마다)
            if do_step:
                # 누적된 기울기로 가중치 수정 + EMA 갱신 (0-9절)
                self.optimizer_step()
                # 마지막 수정 번호 기록
                last_opt_step = ni
                # 시간 제한 학습이면 (우리는 안 씀)
                if self.args.time:
                    # 시간 초과했는지
                    self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                    # 분산 학습이면
                    if RANK != -1:
                        # 0번 GPU의 판단을
                        broadcast_list = [self.stop if RANK == 0 else None]
                        # 모든 GPU에 전달
                        dist.broadcast_object_list(broadcast_list, 0)
                        self.stop = broadcast_list[0]
                    # 시간 초과면 묶음 반복 종료
                    if self.stop:
                        break
            # 메인 프로세스면 진행 막대 갱신
            if RANK in {-1, 0}:
                # 손실 개수 (3)
                loss_length = len(self.tloss)
                # "1/30  5.2G  1.234  0.876  0.987  2133  960" 같은 한 줄
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                    % (
                        # 지금 epoch / 전체
                        f"{epoch + 1}/{self.epochs}",
                        # GPU 메모리 사용량 GB
                        f"{self._get_memory():.3g}G",
                        # 평균 손실 3개
                        *self.tloss.values(),
                        # 이번 묶음 머리 박스 개수
                        batch.get("cls", batch["img"]).shape[0],
                        # 입력 크기
                        batch["img"].shape[-1],
                    )
                )
                # 그림 저장할 묶음이면
                if self.args.plots and ni in self.plot_idx:
                    # 증강된 사진+박스를 train_batch*.jpg로 저장
                    self.plot_training_samples(batch, ni)
            # '묶음 끝' 알림
            self.run_callbacks("on_train_batch_end")
            # 외부에서 멈추라고 했으면
            if self.stop:
                break
        # for가 break 없이 끝났을 때만 실행 (파이썬 for/else)
        else:
            # 첫 epoch을 무사히 마쳤으니 재시도 횟수 초기화
            self._oom_retries = 0
        # 메모리 부족으로 break했으면
        if self._oom_retries and not self.stop:
            # 줄인 batch로 epoch 처음부터
            continue
        # 손실 함수에 epoch마다 바꿀 값이 있으면 갱신 (v8DetectionLoss는 없음)
        if hasattr(unwrap_model(self.model).criterion, "update"):
            unwrap_model(self.model).criterion.update()
        # 이번 epoch 무리별 학습률 기록 (results.csv의 lr/pg0~2)
        self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}
        # 'epoch 끝' 알림
        self.run_callbacks("on_train_epoch_end")
        # 메인 프로세스면
        if RANK in {-1, 0}:
            # EMA 모델에 이름·클래스 수·stride 등 속성 복사 (저장용)
            self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])
        # 마지막 epoch인지
        final_epoch = epoch + 1 >= self.epochs
        # 검증 켜짐(기본) 또는 마지막 또는 곧 멈출 예정이면
        if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
            # GPU 메모리 절반 넘게 쓰면 정리
            self._clear_memory(None if self.device.type == "mps" else 0.5)
            # EMA 가중치로 검증 4,370장 → mAP50·mAP50-95, fitness(= mAP50-95) (0-10절)
            self.metrics, self.fitness = self.validate()
        # 손실이 NaN(숫자 아님)이 되면 마지막 정상 저장본으로 되돌리고 이 epoch을 다시
        if self._handle_nan_recovery(epoch):
            # 수정 주기 초기화
            last_opt_step = -1
            # 망가진 기울기 버림
            self.optimizer.zero_grad()
            # epoch 다시
            continue
        # NaN 복구 횟수 초기화
        self.nan_recovery_attempts = 0
        # 메인 프로세스면
        if RANK in {-1, 0}:
            # results.csv에 한 줄 추가 (학습 손실·검증 지표·학습률)
            self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
            # 조기 종료 검사(patience 100 epoch 동안 fitness 안 오르면) 또는 마지막 epoch이면 멈춤 표시
            self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
            # 시간 제한 학습이면
            if self.args.time:
                # 시간 초과도 멈춤 조건
                self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)
            # 저장 켜짐이면 last.pt 저장, fitness가 최고면 best.pt도 저장
            if (self.args.save or final_epoch) and self.save_model():
                # '모델 저장' 알림
                self.run_callbacks("on_model_save")
        # 지금 시각
        t = time.time()
        # 이번 epoch 걸린 시간
        self.epoch_time = t - self.epoch_time_start
        # 다음 epoch 시작 시각
        self.epoch_time_start = t
        # 시간 제한 학습이면 (우리는 안 씀)
        if self.args.time:
            # 평균 epoch 시간
            mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
            # 남은 시간에 맞게 전체 epoch 수 다시 계산
            self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
            # 워밍업 길이 다시
            nw = self._get_warmup_iterations(nb)
            # 학습률 계획 다시
            self._setup_scheduler()
            # 계획 위치를 지금 epoch으로
            self.scheduler.last_epoch = self.epoch
            # epoch 수를 넘었으면 멈춤
            self.stop |= epoch >= self.epochs
        # 'epoch·검증 끝' 알림
        self.run_callbacks("on_fit_epoch_end")
        # GPU 메모리 절반 넘게 쓰면 정리
        self._clear_memory(None if self.device.type == "mps" else 0.5)
        # 분산 학습이면 멈춤 판단을 모든 GPU에 전달
        if RANK != -1:
            broadcast_list = [self.stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            self.stop = broadcast_list[0]
        # 멈춤이면 epoch 반복 종료
        if self.stop:
            break
        # 다음 epoch
        epoch += 1
    # 총 학습 시간(초)
    seconds = time.time() - self.train_time_start
    # 출력: "30 epochs completed in 3.1 hours."
    LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
    # best.pt로 마지막 검증 한 번 더 (최종 표의 mAP)
    self.final_eval()
    # 메인 프로세스면
    if RANK in {-1, 0}:
        # 그림 켜짐이면
        if self.args.plots:
            # results.png (손실·mAP 곡선) 그리기
            self.plot_metrics()
        # '학습 끝' 알림
        self.run_callbacks("on_train_end")
    # GPU 메모리 정리
    self._clear_memory()
    # 학습·검증 로더마다
    for loader in (self.train_loader, self.test_loader):
        # 닫을 수 있으면
        if hasattr(loader, "close"):
            # 작업 프로세스 종료
            loader.close()
```

## 0-5. 예측: Detect 층과 DFL

```python
# ── nn/modules/head.py  Detect.__init__  (줄 108~143)
# 탐지 머리 만들기. ch = 들어오는 특징 지도들의 채널 수 (YOLO11n: P3 64, P4 128, P5 256) (해설 1·5절)
def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: list[int] | tuple[int, ...] = ()):
    # 부모 초기화
    super().__init__()
    # 클래스 수 (우리 1)
    self.nc = nc
    # 탐지층 개수 = 3 (P2 모델은 4)
    self.nl = len(ch)
    # 박스 한 변 거리를 0~15칸 중 어디인지 16개 확률로 예측 (DFL)
    self.reg_max = reg_max
    # 예측 칸 하나의 출력 수 = 1(클래스) + 16×4(박스) = 65 (COCO는 80+64=144)
    self.no = nc + self.reg_max * 4
    # stride는 모델 생성 때 재서 채움 (0-1절)
    self.stride = torch.zeros(self.nl)
    # 중간 채널 수: c2(박스 갈래) = max(16, 64/4, 64) = 64, c3(클래스 갈래) = max(64, min(1,100)) = 64
    c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
    # cv2 = 박스 갈래, 탐지층마다 하나
    self.cv2 = nn.ModuleList(
        # 3×3 합성곱 → 3×3 합성곱 → 1×1 합성곱으로 64채널(4변 × 16칸) 출력
        nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
    )
    # cv3 = 클래스 갈래 ('머리일 확률'), 탐지층마다 하나
    self.cv3 = (
        # 옛 구조(YOLOv8): 3×3 → 3×3 → 1×1
        nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        if self.legacy
        # YOLO11 구조: 가벼운 합성곱(채널별 3×3 + 1×1) 두 번 → 1×1로 nc채널
        else nn.ModuleList(
            nn.Sequential(
                # 채널별 3×3(DWConv) + 1×1로 c3채널
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                # 한 번 더
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                # 마지막 1×1: c3 → nc(1)채널. 이 층만 COCO에서 못 가져와 새로 학습 (0-1절)
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
    )
    # DFL = 16개 확률 → 거리 하나로 바꾸는 층 (아래 DFL)
    self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    # YOLO26처럼 NMS 없는 모델이면 갈래를 하나 더 (YOLO11은 False)
    if end2end:
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)

# ── nn/modules/head.py  Detect.forward_head  (줄 169~178)
# 특징 지도들을 두 갈래에 통과시켜 모든 예측 칸의 출력을 한 줄로 이어 붙임
def forward_head(
    self, x: list[torch.Tensor], box_head: nn.Module | None = None, cls_head: nn.Module | None = None
) -> dict[str, torch.Tensor]:
    # 갈래가 없으면(추론용 합친 모델) 빈 결과
    if box_head is None or cls_head is None:
        return {}
    # 묶음 안 사진 수
    bs = x[0].shape[0]
    # 박스: 층마다 [사진, 64, 높이×너비]로 펴서 이어 붙임 → [16, 64, 18900] (960 입력: 120²+60²+30²=18,900칸) (해설 5절)
    boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
    # 점수: 층마다 [사진, 1, 높이×너비] → [16, 1, 18900]. 아직 sigmoid 전 값(logit)
    scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
    # 손실 계산에 쓸 세 가지
    return {"boxes": boxes, "scores": scores, "feats": x}

# ── nn/modules/head.py  Detect.forward  (줄 180~204)
# Detect 층 실행
def forward(
    self, x: list[torch.Tensor]
) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    # 기본 갈래(cv2, cv3)로 예측
    preds = self.forward_head(x, **self.one2many)
    # one2one 갈래가 있으면 (YOLO26만)
    if getattr(self, "one2one_cv2", None) is not None:
        # 학습 중엔 기울기가 backbone으로 안 가게 끊음
        x_detach = [xi.detach() for xi in x] if self.training else x
        # 두 번째 갈래 예측
        one2one = self.forward_head(x_detach, **self.one2one)
        # 둘 다 담음
        preds = {"one2many": preds, "one2one": one2one}
    # 학습 중이면 날것 그대로 돌려줌 → 손실 계산으로 (0-6절)
    if self.training:
        return preds
    # 추론 중이면 DFL로 박스 좌표 계산 + sigmoid로 확률 → [사진, 5, 18900]
    y = self._inference(preds["one2one" if self.end2end else "one2many"] if "one2one" in preds else preds)
    # NMS 없는 모델이면 후처리 (YOLO11은 해당 없음 → NMS는 0-10절)
    if self.end2end:
        y = self.postprocess(y.permute(0, 2, 1))
    # 내보내기 모드면 y만, 아니면 (y, 날것)
    return y if self.export else (y, preds)

# ── nn/modules/head.py  Detect.bias_init  (줄 229~241)
# 마지막 층 bias 초기값. 학습 초반이 안정되게 '처음엔 박스 크게, 물체는 거의 없음'으로 시작
def bias_init(self):
    # 탐지층마다 (a = 박스 갈래, b = 클래스 갈래)
    for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
        # 박스 64채널 bias = 2.0
        a[-1].bias.data[:] = 2.0
        # 클래스 bias = log(5 ÷ nc ÷ 칸 수) → P3(640 기준 80×80칸)에서 sigmoid ≈ 0.0008 = '거의 다 배경'
        b[-1].bias.data[: self.nc] = math.log(
            5 / self.nc / (640 / self.stride[i]) ** 2
        )
    # one2one 갈래가 있으면 같은 방식으로 (YOLO11은 없음)
    if getattr(self, "one2one_cv2", None) is not None:
        for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )

# ── nn/modules/block.py  DFL  (줄 60~81)
# DFL: 한 변의 거리를 0,1,…,15칸 각각의 확률로 예측 → 기댓값(확률 × 칸 번호의 합)을 거리로 (해설 5절)
class DFL(nn.Module):
    # c1 = 16칸
    def __init__(self, c1: int = 16):
        # 부모 초기화
        super().__init__()
        # 16 → 1채널 1×1 합성곱, 학습 안 함(가중치 고정)
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        # [0, 1, 2, …, 15]
        x = torch.arange(c1, dtype=torch.float)
        # 합성곱 가중치를 [0…15]로 → 합성곱 = Σ 확률 × 칸 번호 = 기댓값
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        # 16 기억
        self.c1 = c1
    # 실행: x = [사진, 64, 예측 칸 수]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b = 사진 수, a = 예측 칸 수
        b, _, a = x.shape
        # [b, 4변, 16, a]로 나눔 → 16칸 방향 softmax(합 1인 확률) → 기댓값 합성곱 → [b, 4, a] = 칸 단위 거리 4개
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

# ── utils/tal.py  make_anchors  (줄 539~565)
# 모든 예측 칸의 중심 좌표와 stride 목록을 만듦 (해설 4절)
def make_anchors(feats, strides, grid_cell_offset=0.5):
    # 칸 중심 좌표 목록, stride 목록
    anchor_points, stride_tensor = [], []
    # 특징 지도가 꼭 있어야 함
    assert feats is not None
    # 자료형 (16/32비트)
    dtype = feats[0].dtype
    # 탐지층마다 (P3, P4, P5)
    for i in range(len(feats)):
        # 이 층의 stride (8, 16, 32)
        stride = strides[i]
        # 이 층 지도의 높이·너비 (960 입력이면 120×120, 60×60, 30×30)
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        # x 좌표 0.5, 1.5, 2.5, … (칸 한가운데)
        sx = torch.arange(w).type_as(feats[0]) + grid_cell_offset
        # y 좌표 0.5, 1.5, 2.5, …
        sy = torch.arange(h).type_as(feats[0]) + grid_cell_offset
        # 격자로 펼침 (모든 x·y 조합)
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        # [h×w, 2] (x, y) 목록으로 → 지도 칸 단위 좌표
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        # 칸마다 stride 값 [h×w, 1]
        stride_tensor.append(feats[0].new_full((h * w, 1), stride, dtype=dtype))
    # 세 층을 이어 붙임 → [18900, 2], [18900, 1]
    return torch.cat(anchor_points), torch.cat(stride_tensor)

# ── utils/tal.py  dist2bbox  (줄 568~587)
# 칸 중심에서 네 변까지 거리(왼·위·오른·아래) → 박스 좌표
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    # lt = 왼쪽·위쪽 거리, rb = 오른쪽·아래쪽 거리
    lt, rb = distance.chunk(2, dim)
    # 좌상단 = 중심 − (왼, 위)
    x1y1 = anchor_points - lt
    # 우하단 = 중심 + (오른, 아래)
    x2y2 = anchor_points + rb
    # 중심·크기 형식을 원하면
    if xywh:
        # 중심
        c_xy = (x1y1 + x2y2) / 2
        # 너비·높이
        wh = x2y2 - x1y1
        # (cx, cy, w, h)
        return torch.cat([c_xy, wh], dim)
    # 아니면 (x1, y1, x2, y2) — 손실 계산은 이쪽
    return torch.cat((x1y1, x2y2), dim)

# ── utils/tal.py  bbox2dist  (줄 590~605)
# 반대로: 정답 박스 → 칸 중심에서 네 변까지 거리 (DFL 정답 만들 때)
def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int | None = None) -> torch.Tensor:
    # 좌상단, 우하단
    x1y1, x2y2 = bbox.chunk(2, -1)
    # (중심−좌상단, 우하단−중심) = (왼, 위, 오른, 아래) 거리
    dist = torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1)
    # 최대값이 정해졌으면
    if reg_max is not None:
        # 0 ~ 14.99칸으로 자름 → 이보다 큰 머리는 거리를 끝까지 못 배움 (해설 5절)
        dist = dist.clamp_(0, reg_max - 0.01)
    # 거리 4개
    return dist
```

## 0-6. 손실 계산

```python
# ── utils/loss.py  v8DetectionLoss  (줄 355~512)
# YOLO11 탐지 손실 = 박스 위치(box) + 머리 확신(cls) + 거리 분포(dfl) (해설 4·5절)
class v8DetectionLoss:
    # 손실 함수 준비 (모델 만들 때 한 번)
    def __init__(
        self, model: torch.nn.Module, tal_topk: int = 10, tal_topk2: int | None = None
    ):
        # 모델이 있는 장치 (GPU)
        device = next(model.parameters()).device
        # 학습 설정값 (box 7.5, cls 0.5, dfl 1.5 등)
        h = model.args
        # Detect 층
        m = model.model[-1]
        # 이진 교차 엔트로피 (sigmoid 포함), 칸별 값을 그대로 둠(합치지 않음)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        # 설정값 기억
        self.hyp = h
        # stride [8, 16, 32]
        self.stride = m.stride
        # 클래스 수 1
        self.nc = m.nc
        # 칸당 출력 수 65
        self.no = m.nc + m.reg_max * 4
        # 16
        self.reg_max = m.reg_max
        # 장치
        self.device = device
        # DFL 사용 (16 > 1 → True)
        self.use_dfl = m.reg_max > 1
        # 화면에 보이는 손실 이름 3개
        self.loss_names = "box_loss", "cls_loss", "dfl_loss" if self.use_dfl else "l1_loss"
        # 클래스별 가중치 (불균형 데이터용, 기본 없음)
        self.class_weights = getattr(model, "class_weights", None)
        # 있으면
        if self.class_weights is not None:
            # 장치로 옮기고 모양 맞춤
            self.class_weights = self.class_weights.to(device).view(1, 1, -1)
        # 짝짓기 담당: 정답 머리마다 어느 예측 칸이 맡을지 정함 (0-7절)
        self.assigner = TaskAlignedAssigner(
            # 정답 하나당 후보 상위 10개
            topk=tal_topk,
            # 클래스 수 1
            num_classes=self.nc,
            # 점수 = 확신^0.5
            alpha=0.5,
            # × 겹침^6
            beta=6.0,
            # [8, 16, 32]
            stride=self.stride.tolist(),
            # 두 번째 거르기 개수 (없으면 topk와 같음)
            topk2=tal_topk2,
        )
        # 박스 손실 계산기 (0-8절)
        self.bbox_loss = BboxLoss(m.reg_max).to(device)
        # [0, 1, …, 15] (DFL 기댓값 계산용)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)
    # 라벨 정리: [박스 수, 6] 한 줄 목록 → [사진, 그 사진 최대 박스 수, 5] 표로
    def preprocess(self, targets: torch.Tensor, batch_size: int, scale_tensor: torch.Tensor) -> torch.Tensor:
        # nl = 묶음 전체 박스 수, ne = 6 (사진번호, 클래스, x, y, w, h)
        nl, ne = targets.shape
        # 박스가 하나도 없으면
        if nl == 0:
            # 빈 표
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        # 있으면
        else:
            # 각 박스가 몇 번째 사진 것인지
            batch_idx = targets[:, 0].long()
            # 사진마다 박스 개수
            _, counts = batch_idx.unique(return_counts=True)
            # 정수로
            counts = counts.to(dtype=torch.int32)
            # 가장 많은 사진의 박스 수 (CrowdHuman은 한 장에 수백 개 → 메모리 많이 씀)
            n = counts.max()
            # Mac GPU면 2의 거듭제곱으로 올림 (메모리 관리용)
            if self.device.type == "mps":
                n = 1 << (int(n) - 1).bit_length()
            # [사진 수, n, 5] 0으로 채운 표 (빈 자리는 0 = 가짜 박스)
            out = torch.zeros(batch_size, n, ne - 1, device=self.device)
            # 사진마다 시작 위치를 구하기 위한 배열
            offsets = torch.zeros(batch_size + 1, dtype=torch.long, device=self.device)
            # 사진별 박스 수를 해당 칸에 더함
            offsets.scatter_add_(0, batch_idx + 1, torch.ones_like(batch_idx))
            # 누적합 → 사진 k의 첫 박스 위치
            offsets = offsets.cumsum(0)
            # 각 박스가 자기 사진 안에서 몇 번째인지
            within_idx = torch.arange(nl, device=self.device) - offsets[batch_idx]
            # 표에 채워 넣음 (클래스, x, y, w, h)
            out[batch_idx, within_idx] = targets[:, 1:]
            # 0~1 비율 좌표 × 사진 크기 → 픽셀, (중심,크기) → (좌상단,우하단)
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        # [사진, n, 5]
        return out
    # 예측된 64채널 → 박스 좌표 (지도 칸 단위)
    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        # DFL 사용이면
        if self.use_dfl:
            # b = 사진, a = 예측 칸 18,900, c = 64
            b, a, c = pred_dist.shape
            # [b, a, 4변, 16] → softmax → [0…15]와 곱해 기댓값 = 변까지 거리 (DFL 층과 같은 계산)
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_dist.dtype))
        # 거리 → (x1, y1, x2, y2)
        return dist2bbox(pred_dist, anchor_points, xywh=False)
    # 짝짓기와 손실 계산 본체
    def get_assigned_targets_and_loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, Any], **assign_kwargs
    ) -> tuple:
        # 손실 3칸 [box, cls, dfl]
        loss = torch.zeros(3, device=self.device)
        # 예측을 [사진, 18900, 64]와 [사진, 18900, 1]로 축 순서 바꿈
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        # 18,900칸의 중심 좌표(칸 단위)와 stride
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        # 자료형
        dtype = pred_scores.dtype
        # 사진 수 16
        batch_size = pred_scores.shape[0]
        # 입력 크기 = 가장 큰 지도 크기 × 8 = [960, 960]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        # 라벨을 한 줄로 합침: [사진번호, 클래스, x, y, w, h]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        # 사진별 표로 + 픽셀 좌표로 (위 preprocess)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        # 클래스 [사진, n, 1], 박스 [사진, n, 4]
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        # 진짜 박스인지 (빈 자리 0 박스는 False)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        # 예측 박스 좌표 (칸 단위) [사진, 18900, 4]
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        # 짝짓기: 정답 머리마다 담당 예측 칸을 정함 (0-7절). detach = 짝짓기 자체는 학습 안 함
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            # 예측 확신 (0~1)
            pred_scores.detach().sigmoid(),
            # 예측 박스 (픽셀 단위로 바꿈)
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            # 칸 중심 (픽셀 단위)
            anchor_points * stride_tensor,
            # 정답 클래스
            gt_labels,
            # 정답 박스 (픽셀)
            gt_bboxes,
            # 진짜 박스 표시
            mask_gt,
            **assign_kwargs,
        )
        # 정답 점수 합 (나눌 값, 최소 1) — 손실을 '정답 머리 수' 정도로 나눠 크기 맞춤
        target_scores_sum = target_scores.sum().clamp_(min=1)
        # 확신 손실: 18,900칸 전부에 대해 BCE. 담당 칸은 정답 점수(0~1)를, 나머지는 0(배경)을 맞히도록 (해설 4절)
        # → 회색·무시 영역의 칸도 여기서 '배경'으로 배움 (해설 9절에서 빼는 자리)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))
        # 클래스 가중치가 있으면 곱함 (기본 없음)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        # 모두 더해 정답 점수 합으로 나눔
        loss[1] = bce_loss.sum() / target_scores_sum
        # 박스 손실·DFL 손실 (담당 칸만) (0-8절)
        loss[0], loss[2] = self.bbox_loss(
            # 예측 64채널
            pred_distri,
            # 예측 박스 (칸 단위)
            pred_bboxes,
            # 칸 중심 (칸 단위)
            anchor_points,
            # 정답 박스를 칸 단위로
            target_bboxes / stride_tensor,
            # 정답 점수
            target_scores,
            # 나눌 값
            target_scores_sum,
            # 담당 칸 표시
            fg_mask,
            # 입력 크기
            imgsz,
            # stride
            stride_tensor,
        )
        # 박스 손실 × 7.5
        loss[0] *= self.hyp.box
        # 확신 손실 × 0.5
        loss[1] *= self.hyp.cls
        # DFL 손실 × 1.5
        loss[2] *= self.hyp.dfl
        # (짝짓기 결과, 손실, 화면용 손실 사전)
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(self.loss_names, loss.detach())),
        )
    # 모델 출력이 (추론값, 날것) 쌍이면 날것만 꺼냄
    def parse_output(
        self, preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        return preds[1] if isinstance(preds, tuple) else preds
    # 손실 함수를 함수처럼 부를 때 (model(batch) 안에서 불림)
    def __call__(
        self,
        preds: dict[str, torch.Tensor] | tuple[torch.Tensor, dict[str, torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 날것 꺼내서 loss 계산
        return self.loss(self.parse_output(preds), batch)
    # 손실 계산
    def loss(
        self, preds: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # 사진 수 16
        batch_size = preds["boxes"].shape[0]
        # 위 본체에서 손실과 화면용 값만 꺼냄
        loss, loss_detach = self.get_assigned_targets_and_loss(preds, batch)[1:]
        # 손실 × 사진 수 (기울기 크기를 batch 크기에 비례하게) → backward로 (0-4절)
        return loss * batch_size, loss_detach
```

## 0-7. 짝짓기

```python
# ── utils/tal.py  TaskAlignedAssigner  (줄 14~393)
# 정답 머리마다 18,900개 예측 칸 중 '맡을 칸'을 고르는 규칙 (해설 4절)
class TaskAlignedAssigner(nn.Module):
    # 설정값 받기 (v8DetectionLoss가 topk=10, alpha=0.5, beta=6으로 넘김)
    def __init__(
        self,
        # 정답 하나당 후보 수
        topk: int = 13,
        # 클래스 수
        num_classes: int = 80,
        # 확신의 지수
        alpha: float = 1.0,
        # 겹침의 지수
        beta: float = 6.0,
        # stride 목록
        stride: list | None = None,
        # 0으로 나누기 방지용 아주 작은 수
        eps: float = 1e-9,
        # 두 번째 거르기 개수
        topk2: int | None = None,
    ):
        # 부모 초기화
        super().__init__()
        # 10
        self.topk = topk
        # 10 (따로 안 주면 topk와 같음)
        self.topk2 = topk2 or topk
        # 1
        self.num_classes = num_classes
        # 0.5
        self.alpha = alpha
        # 6
        self.beta = beta
        # [8, 16, 32]
        self.stride = stride if stride is not None else [8, 16, 32]
        # 두 번째 stride = 16 → 작은 머리를 16px로 넓혀 볼 때 씀 (아래 select_candidates_in_gts)
        self.stride_val = self.stride[1] if len(self.stride) > 1 else self.stride[0]
        # 1e-9
        self.eps = eps
        # 메모리 부족 경고를 이미 했는지
        self._oom_warned = False
    # 기울기 계산 안 함 (짝짓기는 학습 대상이 아님)
    @torch.no_grad()
    # 실행: 예측 점수·박스, 칸 중심, 정답 클래스·박스, 진짜 박스 표시
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **extra):
        # 사진 수, (칸 수), 클래스 수
        self.bs, _, self.num_classes = pd_scores.shape
        # 사진당 최대 정답 박스 수
        self.n_max_boxes = gt_bboxes.shape[1]
        # 정답이 하나도 없으면 (머리 없는 사진만 모인 묶음)
        if self.n_max_boxes == 0:
            # 전부 배경: 클래스=없음, 박스 0, 점수 0, 담당 없음
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                torch.zeros_like(pd_scores[..., 0]),
            )
        # 보통은 한 번에 계산
        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **extra)
        # 오류면
        except RuntimeError as e:
            # 메모리 부족이 아니면 그대로 오류
            if "out of memory" not in str(e).lower():
                raise
        # 메모리 부족이면: [사진 × 머리 수 × 18,900] 표가 너무 큼 (머리 200개 사진) → 사진 한 장씩 다시
        bs, n_max_boxes = self.bs, self.n_max_boxes
        # 처음 한 번만 경고
        if not self._oom_warned:
            LOGGER.warning(
                f"CUDA out of memory in TaskAlignedAssigner with batch_size={bs} and max_num_obj={n_max_boxes}; "
                "retrying assignment one image at a time on GPU. Model forward batch size is unchanged."
            )
            self._oom_warned = True
        # 사진마다 실제 마지막 정답 번호 (빈 자리 빼고 필요한 만큼만)
        last_gt_idx = (
            # [사진, n] 진짜 박스 표시
            mask_gt.squeeze(-1)
            .bool()
            # 진짜면 1, 2, 3, … 번호
            .mul(torch.arange(1, n_max_boxes + 1, device=mask_gt.device))
            # 사진마다 가장 큰 번호 = 박스 수
            .amax(1)
            # 최소 1
            .clamp_(min=1)
            # 파이썬 목록으로
            .tolist()
        )
        # 한 장씩
        self.bs = 1
        # 결과 담을 곳
        results = None
        try:
            # 사진 i, 그 사진의 박스 수
            for i, self.n_max_boxes in enumerate(last_gt_idx):
                # 그 사진만 잘라 짝짓기
                result = self._forward(
                    pd_scores[i : i + 1],
                    pd_bboxes[i : i + 1],
                    anc_points,
                    gt_labels[i : i + 1, : self.n_max_boxes],
                    gt_bboxes[i : i + 1, : self.n_max_boxes],
                    mask_gt[i : i + 1, : self.n_max_boxes],
                    **{k: v[i : i + 1] for k, v in extra.items()},
                )
                # 처음이면 전체 결과 그릇 만들기
                if results is None:
                    results = tuple(x.new_empty((bs, *x.shape[1:])) for x in result)
                # i번째 자리에 넣음
                for output, x in zip(results, result):
                    output[i] = x[0]
        # 끝나면 원래 값 복원
        finally:
            self.bs, self.n_max_boxes = bs, n_max_boxes
        # 모은 결과
        return results
    # 짝짓기 본체
    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, **extra):
        # ①②③ 후보 고르기 → 점수 계산 → 상위 10개 (아래 get_pos_mask)
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, **extra
        )
        # ④ 한 칸이 여러 머리에 뽑혔으면 하나만 남김
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes, align_metric
        )
        # 칸마다 맡은 머리의 클래스·박스·점수(0/1)를 꺼냄
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
        # 담당이 아닌 칸의 점수는 0으로
        align_metric *= mask_pos
        # 머리마다 담당 칸 중 가장 높은 점수
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        # 담당 아닌 칸의 겹침도 0으로
        overlaps *= mask_pos
        # 머리마다 담당 칸 중 가장 높은 겹침(CIoU)
        pos_overlaps = overlaps.amax(dim=-1, keepdim=True)
        # 점수를 '그 머리의 최고 점수 = 그 머리의 최고 겹침'이 되게 비례 조정
        align_metric.mul_(pos_overlaps).div_(pos_align_metrics + self.eps)
        # 칸마다(여러 머리 중) 최댓값 → 그 칸의 정답 점수 (0~1)
        norm_align_metric = align_metric.amax(-2).unsqueeze(-1)
        # 정답 점수 = 1 × 위 값 → 박스가 잘 맞을수록 1에 가까움. 그래서 학습된 확신이 1까지 잘 안 오름 → 인원 셀 때 기준 0.15 (해설 4절)
        target_scores = target_scores * norm_align_metric
        # 클래스, 정답 박스, 정답 점수, 담당 여부, 맡은 머리 번호
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
    # 담당 후보 고르기 (①②③)
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, **extra):
        # ① 칸 중심이 정답 박스 안에 있는 칸만 [사진, 머리, 18900]
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, mask_gt)
        # ② 후보 칸마다 점수 = 확신^0.5 × CIoU^6
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt, **extra
        )
        # ③ 머리마다 점수 상위 10칸
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # 세 조건을 모두 만족하는 칸 = 담당 후보 (상위 10 AND 박스 안 AND 진짜 머리)
        mask_pos = mask_topk.mul_(mask_in_gts).mul_(mask_gt.bool())
        # 후보, 점수, 겹침
        return mask_pos, align_metric, overlaps
    # ② 점수 계산
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt, **extra):
        # 예측 칸 수 18,900
        na = pd_bboxes.shape[-2]
        # 후보 표시 (참/거짓)
        mask_gt = mask_gt.bool()
        # 결과 표 모양 [사진, 머리, 18900]
        shape = self.bs, self.n_max_boxes, na
        # 후보인 (사진, 머리, 칸) 위치 목록 → 후보만 계산해 메모리 절약
        indices = mask_gt.nonzero(as_tuple=True)
        # 그 칸이 그 머리의 클래스(head)를 얼마나 확신하는지
        bbox_scores = pd_scores[indices[0], indices[2], gt_labels[indices[0], indices[1], 0].long()]
        # 그 칸의 예측 박스와 정답 박스의 겹침(CIoU)
        overlap_values = self.pair_overlaps(indices, gt_bboxes, pd_bboxes, **extra)
        # 점수 = 확신^0.5 × 겹침^6 → 겹침 0.9면 0.53, 0.7이면 0.12: 박스가 조금만 어긋나도 점수 급락
        align_values = bbox_scores.pow(self.alpha) * overlap_values.pow(self.beta)
        # 겹침 표 (후보 아닌 곳 0)
        overlaps = torch.zeros(shape, dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        # 점수 표 (후보 아닌 곳 0)
        align_metric = torch.zeros(shape, dtype=align_values.dtype, device=pd_scores.device)
        # 후보 자리에 겹침 채움
        overlaps[indices] = overlap_values
        # 후보 자리에 점수 채움
        align_metric[indices] = align_values
        # 점수 표, 겹침 표
        return align_metric, overlaps
    # 후보 쌍의 겹침 계산
    def pair_overlaps(self, indices, gt_bboxes, pd_bboxes):
        # 정답 박스와 예측 박스를 짝지어 iou_calculation
        return self.iou_calculation(gt_bboxes[indices[:2]], pd_bboxes[indices[0], indices[2]])
    # 겹침 = CIoU (IoU에 중심 거리·가로세로 비율 차이 벌점), 음수는 0으로
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)
    # ③ 머리마다 점수 상위 topk칸 고르기
    def select_topk_candidates(self, metrics, topk_mask=None):
        # 머리마다 점수 큰 순서로 10개의 값과 칸 번호
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        # 표시가 없으면 (여기선 항상 넘겨받음)
        if topk_mask is None:
            # 최고 점수가 0보다 큰 머리만
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # 가짜 머리(빈 자리)의 칸 번호는 0으로
        topk_idxs.masked_fill_(~topk_mask, 0)
        # 칸마다 몇 번 뽑혔는지 세는 표
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        # 뽑힌 칸에 1씩 더함
        count_tensor.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
        # 한 머리 안에서 2번 이상 = 가짜 머리의 0번 칸 중복 → 지움
        count_tensor.masked_fill_(count_tensor > 1, 0)
        # [사진, 머리, 18900] 0/1 표
        return count_tensor
    # 칸마다 맡은 머리의 정답 꺼내기
    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        # 사진 번호 [사진, 1]
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        # (사진 안 머리 번호) → (묶음 전체에서 몇 번째 머리) 로 바꿈
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        # 칸마다 맡은 머리의 클래스
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        # 칸마다 맡은 머리의 박스 [사진, 18900, 4]
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]
        # 음수 클래스 방지
        target_labels.clamp_(0)
        # 원-핫 표 [사진, 18900, 클래스 수] (F.one_hot보다 빠름)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int8,
            device=target_labels.device,
        )
        # 맡은 클래스 자리에 1
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        # 담당이 아닌 칸은 전부 0 (배경)
        target_scores = target_scores * (fg_mask[:, :, None] > 0)
        # 클래스, 박스, 점수(0/1)
        return target_labels, target_bboxes, target_scores
    # ① 칸 중심이 정답 박스 안에 있는지
    def select_candidates_in_gts(self, xy_centers, gt_bboxes, mask_gt, eps=1e-9):
        # 정답 박스를 (중심, 크기) 형식으로
        gt_bboxes_xywh = xyxy2xywh(gt_bboxes)
        # 너비·높이가 16px보다 작은지
        wh_mask = gt_bboxes_xywh[..., 2:] < self.stride_val
        # 작은 변은 16px로 넓혀 봄 → 8px 머리도 박스 안 칸 중심이 생겨 담당이 0개가 안 됨 (해설 4절)
        gt_bboxes_xywh[..., 2:] = torch.where(
            # 작고 진짜인 박스면
            (wh_mask * mask_gt).bool(),
            # 16으로
            torch.tensor(self.stride_val, dtype=gt_bboxes_xywh.dtype, device=gt_bboxes_xywh.device),
            # 아니면 그대로
            gt_bboxes_xywh[..., 2:],
        )
        # 다시 (좌상단, 우하단) 형식으로
        gt_bboxes = xywh2xyxy(gt_bboxes_xywh)
        # lt = 좌상단, rb = 우하단 [사진, 머리, 1, 2]
        lt, rb = gt_bboxes.unsqueeze(2).chunk(2, 3)
        # 칸 중심 x가 왼쪽 변보다 오른쪽
        mask = xy_centers[:, 0] - lt[..., 0] > eps
        # 칸 중심 y가 위쪽 변보다 아래
        mask &= xy_centers[:, 1] - lt[..., 1] > eps
        # 칸 중심 x가 오른쪽 변보다 왼쪽
        mask &= rb[..., 0] - xy_centers[:, 0] > eps
        # 칸 중심 y가 아래쪽 변보다 위
        mask &= rb[..., 1] - xy_centers[:, 1] > eps
        # 네 조건 모두 = 박스 안 [사진, 머리, 18900]
        return mask
    # ④ 한 칸이 여러 머리의 후보가 되면 겹침(CIoU)이 가장 큰 머리 하나만
    def select_highest_overlaps(self, mask_pos, overlaps, n_max_boxes, align_metric):
        # 칸마다 몇 개 머리의 후보인지 [사진, 18900]
        fg_mask = mask_pos.sum(-2)
        # 2개 이상인 칸 표시를 머리 축으로 펼침 [사진, 머리, 18900] (빽빽한 군중에서 자주 생김)
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
        # 칸마다 겹침이 가장 큰 머리 번호
        max_overlaps_idx = overlaps.max(1).indices
        # 그 머리 자리만 1인 표
        is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        # 겹친 칸은 '최대 겹침 머리'만, 나머지 칸은 그대로
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        # topk2가 topk와 다르면 한 번 더 거름 (우리는 같아서 건너뜀)
        if self.topk2 != self.topk:
            # 남은 후보의 점수만
            align_metric = align_metric * mask_pos
            # 머리마다 상위 topk2칸
            max_overlaps_idx = torch.topk(align_metric, self.topk2, dim=-1, largest=True).indices
            # 그 칸만 1인 표
            topk_idx = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            topk_idx.scatter_(-1, max_overlaps_idx, 1.0)
            # 겹치는 것만 남김
            mask_pos *= topk_idx
        # 머리 축 최댓값 → fg_mask(담당 여부 0/1), target_gt_idx(맡은 머리 번호) (화웨이 NPU면 int64로)
        fg_mask, target_gt_idx = (mask_pos.to(torch.int64) if mask_pos.device.type == "npu" else mask_pos).max(-2)
        # 맡은 머리 번호, 담당 여부, 최종 후보 표
        return target_gt_idx, fg_mask, mask_pos
```

## 0-8. 박스 손실과 DFL 손실

```python
# ── utils/loss.py  BboxLoss  (줄 119~164)
# 담당 칸만 대상으로 박스 위치 손실(CIoU)과 거리 분포 손실(DFL)을 계산 (해설 5절)
class BboxLoss(nn.Module):
    # reg_max = 16
    def __init__(self, reg_max: int = 16):
        # 부모 초기화
        super().__init__()
        # DFL 손실 계산기 (16 > 1이라 사용)
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
    # 실행
    def forward(
        self,
        # 예측 64채널 [사진, 18900, 64]
        pred_dist: torch.Tensor,
        # 예측 박스 (칸 단위)
        pred_bboxes: torch.Tensor,
        # 칸 중심 (칸 단위)
        anchor_points: torch.Tensor,
        # 정답 박스 (칸 단위)
        target_bboxes: torch.Tensor,
        # 정답 점수 (0~1)
        target_scores: torch.Tensor,
        # 정답 점수 합
        target_scores_sum: torch.Tensor,
        # 담당 칸 표시
        fg_mask: torch.Tensor,
        # 입력 크기 (DFL 안 쓸 때만)
        imgsz: torch.Tensor,
        # stride (DFL 안 쓸 때만)
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 담당 칸 위치 목록 (한 번만 계산)
        fg_mask = fg_mask.nonzero(as_tuple=True)
        # 담당 칸마다 무게 = 그 칸의 정답 점수 → 박스가 잘 맞는 담당 칸일수록 손실에 크게 반영
        weight = target_scores[fg_mask].sum(-1, keepdim=True)
        # 담당 칸의 예측 박스와 정답 박스의 CIoU
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        # 박스 손실 = Σ (1 − CIoU) × 무게 ÷ 정답 점수 합
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        # DFL을 쓰면 (YOLO11)
        if self.dfl_loss:
            # 정답 박스 → 칸 중심에서 네 변까지 거리, 최대 14.99칸 (stride 8이면 약 120px)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            # 담당 칸의 16개 확률 [칸×4, 16]과 정답 거리로 DFL 손실 × 무게
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            # 합 ÷ 정답 점수 합
            loss_dfl = loss_dfl.sum() / target_scores_sum
        # DFL 안 쓰면 (YOLO26 등) 거리 차이(L1)로 대신
        else:
            # 정답 거리 (칸 단위)
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            # 픽셀 단위로
            target_ltrb = target_ltrb * stride
            # 가로 거리는 너비로 나눠 0~1로
            target_ltrb[..., 0::2] /= imgsz[1]
            # 세로 거리는 높이로 나눠 0~1로
            target_ltrb[..., 1::2] /= imgsz[0]
            # 예측도 픽셀 단위로
            pred_dist = pred_dist * stride
            # 가로 0~1
            pred_dist[..., 0::2] /= imgsz[1]
            # 세로 0~1
            pred_dist[..., 1::2] /= imgsz[0]
            # 네 거리 차이의 평균 × 무게
            loss_dfl = (
                F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
            )
            # 합 ÷ 정답 점수 합
            loss_dfl = loss_dfl.sum() / target_scores_sum
        # 박스 손실, DFL 손실 (위에서 × 7.5, × 1.5)
        return loss_iou, loss_dfl

# ── utils/loss.py  DFLoss  (줄 97~116)
# DFL 손실: 정답 거리가 3.3칸이면 '3칸 확률 0.7 + 4칸 확률 0.3'이 되도록 학습 (해설 5절)
class DFLoss(nn.Module):
    # 16칸
    def __init__(self, reg_max: int = 16) -> None:
        # 부모 초기화
        super().__init__()
        # 16
        self.reg_max = reg_max
    # pred_dist = [N, 16] 예측(아직 softmax 전), target = [N] 정답 거리
    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 정답 거리를 0 ~ 14.99로
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        # 왼쪽 정수 칸 (3.3 → 3)
        tl = target.long()
        # 오른쪽 정수 칸 (4)
        tr = tl + 1
        # 왼쪽 칸 무게 = 4 − 3.3 = 0.7
        wl = tr - target
        # 오른쪽 칸 무게 = 0.3
        wr = 1 - wl
        # 16칸 로그 확률 (log softmax)
        logp = F.log_softmax(pred_dist, dim=1)
        # 손실 = −(0.7 × log P(3칸) + 0.3 × log P(4칸)), 네 변 평균
        return -(
            logp.gather(1, tl.view(-1, 1)).view(tl.shape) * wl + logp.gather(1, tr.view(-1, 1)).view(tl.shape) * wr
        ).mean(-1, keepdim=True)
```

## 0-9. 가중치 고치기와 EMA

```python
# ── engine/trainer.py  BaseTrainer.optimizer_step  (줄 925~936)
# 4묶음(유효 batch 64)마다 한 번: 누적된 기울기로 가중치를 고침 (해설 6절)
def optimizer_step(self):
    # AMP 때 키워 둔 기울기를 원래 크기로 되돌림
    self.scaler.unscale_(self.optimizer)
    # 화웨이 NPU면 (우리는 아님)
    if self.device.type == "npu" and TORCH_2_0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0, foreach=False)
    # 보통
    else:
        # 전체 기울기 크기(노름)가 10을 넘으면 10이 되게 줄임 → 한 번에 너무 크게 고치지 않게
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
    # AdamW로 가중치 수정 (기울기에 inf·NaN이 있으면 이번 수정은 건너뜀)
    self.scaler.step(self.optimizer)
    # 다음 번 손실 확대 배율 조정
    self.scaler.update()
    # 기울기 0으로 → 다음 4묶음 누적 시작
    self.optimizer.zero_grad()
    # EMA가 있으면
    if self.ema:
        # 평균 가중치 갱신 (아래)
        self.ema.update(self.model)

# ── utils/torch_utils.py  ModelEMA  (줄 817~892)
# EMA: 가중치의 '이동 평균' 사본. 고칠 때마다 조금씩 따라가 흔들림이 적음. 검증·best.pt는 이 사본 (해설 6절)
class ModelEMA:
    # decay 0.9999, tau 2000
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # 모델을 통째로 복사해 평가 모드로
        self.ema = deepcopy(unwrap_model(model)).eval()
        # 지식 증류 모델이면 (우리는 아님)
        if hasattr(self.ema, "teacher_model"):
            # 선생 모델은 사본에서 빼서 메모리 절약
            self.ema.teacher_model = None
        # 지금까지 갱신 횟수 (resume이면 이어서)
        self.updates = updates
        # 따라가는 정도 d = 0.9999 × (1 − e^(−횟수/2000)) → 처음엔 0에 가까워 빨리 따라가고, 수천 번 뒤엔 0.9999로 천천히
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        # 사본의 모든 파라미터에
        for p in self.ema.parameters():
            # 기울기 계산 안 함 (학습 대상 아님)
            p.requires_grad_(False)
        # 사용 중
        self.enabled = True
        # (사본 텐서, 모델 텐서) 짝 목록, 첫 갱신 때 만듦
        self._pairs = None
    # 갱신 (optimizer_step 끝에서)
    def update(self, model):
        # 사용 중이면
        if self.enabled:
            # 횟수 +1
            self.updates += 1
            # 이번 d (예: 100번째 0.049, 2000번째 0.632, 7050번째 0.970)
            d = self.decay(self.updates)
            # 짝 목록이 없으면 (첫 갱신)
            if self._pairs is None:
                # 학습 중인 모델의 가중치 사전
                msd = unwrap_model(model).state_dict()
                # 사본 쪽, 모델 쪽 목록
                ema_v, model_v = [], []
                # 사본의 모든 텐서마다
                for k, v in self.ema.state_dict().items():
                    # 실수 텐서만 (정수인 BN 횟수 등 제외)
                    if v.dtype.is_floating_point:
                        # 사본 텐서
                        ema_v.append(v)
                        # 같은 이름의 모델 텐서
                        model_v.append(msd[k])
                # 기억 (텐서는 제자리에서 바뀌므로 한 번만 만들면 됨)
                self._pairs = ema_v, model_v
            # 짝 목록 꺼냄
            ema_v, model_v = self._pairs
            # PyTorch 2.0 이상이고 NPU·옛 Mac이 아니면
            if (
                ema_v and TORCH_2_0 and ema_v[0].device.type != "npu" and (TORCH_2_4 or ema_v[0].device.type != "mps")
            ):
                # 모든 텐서를 한 번에: 사본 = 사본 + (1−d) × (모델 − 사본) = d × 사본 + (1−d) × 모델
                torch._foreach_lerp_(ema_v, model_v, 1 - d)
            # 아니면 하나씩 같은 계산
            else:
                for v, m in zip(ema_v, model_v):
                    # 사본 × d + 모델 × (1−d)
                    v.mul_(d).add_(m, alpha=1 - d)
    # 가중치 말고 속성(이름·stride 등)을 사본에 복사 (epoch 끝마다)
    def update_attr(self, model, include=(), exclude=()):
        # 사용 중이면
        if self.enabled:
            # include 목록의 속성만 복사
            copy_attr(self.ema, unwrap_model(model), include, exclude)
```

## 0-10. 검증: NMS와 best.pt 기준

```python
# ── utils/nms.py  non_max_suppression  (줄 13~157)
# NMS: 같은 머리에 겹친 박스 여러 개 중 확신이 가장 큰 것 하나만 남김 (해설 7절)
def non_max_suppression(
    # 모델 출력 [사진, 4+클래스, 18900]
    prediction,
    # 확신 기준 (검증 때는 0.001로 거의 다 남김, 인원 셀 때 우리는 0.15)
    conf_thres: float = 0.25,
    # 겹침 기준 (검증 때 0.7: 이보다 많이 겹치면 같은 머리로 보고 지움)
    iou_thres: float = PREDICT_IOU,
    # 특정 클래스만 남기기
    classes=None,
    # 클래스 구분 없이 NMS
    agnostic: bool = False,
    # 한 박스에 여러 클래스 허용
    multi_label: bool = False,
    # 한 장에 남길 최대 박스 수 300 → 머리가 300개 넘는 사진은 잘림. 우리 평가는 1,000으로 늘림
    max_det: int = 300,
    # 클래스 수 (안 주면 모양에서 계산)
    nc: int = 0,
    # 사진당 시간 한도 (초)
    max_time_img: float = 0.05,
    # NMS에 넣을 최대 후보 수
    max_nms: int = 30000,
    # 클래스별로 박스를 떨어뜨려 놓을 거리 (픽셀)
    max_wh: int = 7680,
    # 회전 박스 여부
    rotated: bool = False,
    # NMS 없는 모델 여부
    end2end: bool = False,
    # 남은 박스의 원래 번호도 돌려줄지
    return_idxs: bool = False,
):
    # 확신 기준이 0~1인지 확인
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    # 겹침 기준이 0~1인지 확인
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    # 검증 모드 출력은 (추론값, 날것) 쌍이므로
    if isinstance(prediction, (list, tuple)):
        # 추론값만
        prediction = prediction[0]
    # 클래스 거르기가 있으면
    if classes is not None:
        # 텐서로
        classes = torch.tensor(classes, device=prediction.device)
    # NMS 없는 모델(YOLO26)이면 확신 기준만 적용 (YOLO11은 해당 없음)
    if prediction.shape[-1] == 6 or end2end:
        # 결과, 번호
        output, keepi = [], []
        # 사진마다
        for pred in prediction:
            # 확신 기준 넘는 것
            mask = pred[:, 4] > conf_thres
            # 클래스 거르기
            if classes is not None:
                mask &= (pred[:, 5:6] == classes).any(1)
            # 앞에서 max_det개까지
            idx = mask.nonzero(as_tuple=False).view(-1)[:max_det]
            # 32비트로
            output.append(pred[idx].float())
            # 번호
            keepi.append(idx)
        # 돌려줌
        return (output, keepi) if return_idxs else output
    # 사진 수
    bs = prediction.shape[0]
    # 클래스 수 = 5 − 4 = 1
    nc = nc or (prediction.shape[1] - 4)
    # 추가 정보 수 (분할 마스크 등, 탐지는 0)
    extra = prediction.shape[1] - nc - 4
    # 추가 정보 시작 위치 = 5
    mi = 4 + nc
    # 칸마다 '확신이 기준보다 큰가' [사진, 18900]
    xc = prediction[:, 4:mi].amax(1) > conf_thres
    # 칸 원래 번호 [사진, 18900, 1]
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]
    # 시간 한도 = 2초 + 0.05 × 사진 수
    time_limit = 2.0 + max_time_img * bs
    # 클래스가 1개면 다중 라벨 의미 없음 → False
    multi_label &= nc > 1
    # [사진, 18900, 5]로 축 바꿈 (한 줄 = 박스 하나)
    prediction = prediction.transpose(-1, -2)
    # 시작 시각
    t = time.time()
    # 사진마다 빈 결과 [0, 6] (x1,y1,x2,y2,확신,클래스)
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    # 사진마다 빈 번호 목록
    keepi = [torch.empty(0, dtype=torch.long, device=prediction.device)] * bs
    # 사진마다: xi = 번호, x = 그 사진 예측, xk = 칸 번호
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        # 확신 기준 통과 표시
        filt = xc[xi]
        # 통과한 칸만 남김 (18,900 → 수백 개)
        x = x[filt]
        # 회전 박스가 아니면
        if not rotated:
            # (중심, 크기) → (좌상단, 우하단)
            x[:, :4] = xywh2xyxy(x[:, :4])
        # 번호도 필요하면
        if return_idxs:
            # 같이 거름
            xk = xk[filt]
        # 남은 게 없으면
        if not x.shape[0]:
            # 다음 사진
            continue
        # 박스 4, 클래스 확신 nc, 추가 정보로 나눔
        box, cls, mask = x.split((4, nc, extra), 1)
        # 다중 라벨이면 (우리는 False)
        if multi_label:
            # 기준 넘는 (박스, 클래스) 쌍 모두
            i, j = torch.where(cls > conf_thres)
            # 쌍마다 한 줄
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            # 번호
            if return_idxs:
                xk = xk[i]
        # 보통: 박스마다 가장 확신 큰 클래스 하나 (우리는 head뿐)
        else:
            # 최대 확신과 그 클래스
            conf, j = cls.max(1, keepdim=True)
            # 기준 넘는지
            filt = conf.view(-1) > conf_thres
            # [x1, y1, x2, y2, 확신, 클래스] 한 줄로 만들고 거름
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
            # 번호
            if return_idxs:
                xk = xk[filt]
        # 클래스 거르기가 있으면
        if classes is not None:
            # 해당 클래스만
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
        # 남은 박스 수
        n = x.shape[0]
        # 없으면
        if not n:
            # 다음 사진
            continue
        # 3만 개가 넘으면
        if n > max_nms:
            # 확신 큰 순서로 3만 개만
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]
        # 클래스 번호 × 7680 → 클래스가 다르면 박스를 멀리 떨어뜨려 서로 안 지우게 (우리는 클래스 1개라 0)
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        # 확신
        scores = x[:, 4]
        # 클래스만큼 옮긴 박스
        boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1) if rotated else x[:, :4] + c
        # NMS 실행: 확신 큰 박스부터 고르고, 그 박스와 IoU 0.7 넘게 겹치는 박스는 지움 → 남은 번호
        # (빽빽한 군중에서 진짜 다른 두 머리가 0.7 넘게 겹치면 하나가 지워짐)
        i = nms_boxes(boxes, scores, iou_thres, rotated=rotated)
        # 최대 max_det개까지
        i = i[:max_det]
        # 이 사진의 최종 박스
        output[xi] = x[i]
        # 번호도
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        # 시간 한도를 넘으면
        if (time.time() - t) > time_limit:
            # 경고하고
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            # 남은 사진은 빈 결과로 끝냄
            break
    # 사진마다 [박스 수, 6]
    return (output, keepi) if return_idxs else output

# ── utils/metrics.py  Metric.fitness  (줄 1013~1016)
# best.pt를 고르는 점수: epoch마다 이 값이 최고면 best.pt로 저장 (해설 7절)
def fitness(self) -> float:
    # 정밀도·재현율·mAP50은 무게 0, mAP50-95만 1 → best.pt 기준 = mAP50-95 (Ultralytics 8.4.208)
    w = [0.0, 0.0, 0.0, 1.0]
    # [P, R, mAP50, mAP50-95] × 무게의 합 = mAP50-95 (NaN은 0으로)
    return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())
```
