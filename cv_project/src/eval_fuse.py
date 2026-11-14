"""전신+머리 결합 평가: 전신만 / 머리만 / 결합 — 인원 오차와 '진짜 사람을 맞힌 비율'.

usage: python src/eval_fuse.py crowdhuman [최대장수]   CrowdHuman 검증 4,370장 (head_base + yolo11n)
       python src/eval_fuse.py ht21                     HT21-03 1,000프레임 (head_base, head_ht21 + yolo11n 입력 1920)  ← crowdhuman 먼저
출력 results/eval_fuse/crowdhuman.json, ht21.json

공정하게 비교하는 법 (eval_tiles.py와 같은 원칙)
  결합은 머리만 셀 때보다 항상 같거나 많이 센다. 그래서 전신 기준(0.3·0.5·0.7)마다 '센 인원 합 = 실제 인원 합'이 되는
  머리 기준을 다시 찾고, 그 상태에서 맞힌 사람 비율(재현율)과 인원 오차를 비교한다.
맞힌 사람 판정 (정답은 머리 박스뿐이라 이렇게 정함)
  - 머리가 있는 예측(짝·머리만): 머리 박스가 정답 머리와 IoU ≥ 0.5
  - 전신만 잡힌 예측: 아직 안 맞힌 정답 머리의 중심이 그 전신 박스 위쪽 1/3 안에 있으면 맞힘 (결합 규칙과 같은 기준)
"""
import sys, json, pathlib as _pl
import numpy as np
ROOT = _pl.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cvio import imread
from fuse_count import fuse
from eval_heads import load_gt, matched, CROWD_BINS
from ultralytics import YOLO
OUT = ROOT / "results" / "eval_fuse"
HCONFS = np.round(np.arange(0.05, 0.71, 0.01), 2)
BCONFS = (0.3, 0.5, 0.7)


def detect(model, imgs, imgsz, conf, cls=None):
    out = []
    for p in imgs:
        r = model.predict(imread(p), imgsz=imgsz, conf=conf, classes=cls, max_det=1000, verbose=False)[0]
        out.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()))
    return out


def fused(hd, bd, th, tb):
    """한 장: 기준 적용 → (머리 박스, 머리 신뢰도, 전신 박스, 짝, 머리만, 전신만)"""
    H, hc = hd[0][hd[1] >= th], hd[1][hd[1] >= th]
    B = bd[0][bd[1] >= tb] if tb is not None else np.zeros((0, 4))
    return (H, hc, B) + fuse(H, B)


def counts(heads, bodies, th, tb):
    return np.array([(lambda f: len(f[3]) + len(f[4]) + len(f[5]))(fused(h, b, th, tb)) for h, b in zip(heads, bodies)])


def calibrate(heads, bodies, tb, gt_n):
    """센 합 = 실제 합에 가장 가까운 머리 기준. 머리 기준을 낮추면 인원은 줄지 않으므로(단조) 이분 탐색."""
    lo, hi = 0, len(HCONFS) - 1
    while lo < hi:                                   # 합이 실제 이하가 되는 가장 낮은 기준
        mid = (lo + hi) // 2
        if counts(heads, bodies, HCONFS[mid], tb).sum() <= gt_n.sum(): hi = mid
        else: lo = mid + 1
    cand = [i for i in (lo - 1, lo) if i >= 0]
    return float(HCONFS[min(cand, key=lambda i: abs(counts(heads, bodies, HCONFS[i], tb).sum() - gt_n.sum()))])


def hits(g, f):
    """정답 머리 g 중 맞힌 수 (위 '맞힌 사람 판정')"""
    H, hc, B, pairs, ho, bo = f
    hit = matched(g, H, hc)
    if len(bo) and len(g):
        cx = (g[:, 0] + g[:, 2]) / 2; cy = (g[:, 1] + g[:, 3]) / 2
        for b in B[bo]:
            c = np.nonzero(~hit & (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[1] + (b[3] - b[1]) / 3))[0]
            if len(c):
                hit[c[np.argmin(np.hypot(cx[c] - (b[0] + b[2]) / 2, cy[c] - b[1]))]] = True
    return int(hit.sum())


def stats(heads, bodies, gts, th, tb, crowd=True):
    gt_n = np.array([len(g) for g in gts])
    fs = [fused(h, b, th, tb) for h, b in zip(heads, bodies)]
    c = np.array([len(f[3]) + len(f[4]) + len(f[5]) for f in fs]); tp = sum(hits(g, f) for g, f in zip(gts, fs))
    s = {"head_conf": th, "body_conf": tb, "MAE": float(np.abs(c - gt_n).mean()), "found_ratio": float(c.sum() / gt_n.sum()),
         "recall": tp / gt_n.sum(), "precision": tp / max(c.sum(), 1),
         "parts": {"pair": int(sum(len(f[3]) for f in fs)), "head_only": int(sum(len(f[4]) for f in fs)), "body_only": int(sum(len(f[5]) for f in fs))}}
    if crowd:
        s["by_crowd"] = [{"range": f"{lo}-{hi if hi < 10**9 else ''}", "ratio": float(c[(gt_n >= lo) & (gt_n <= hi)].sum() / max(gt_n[(gt_n >= lo) & (gt_n <= hi)].sum(), 1)),
                          "MAE": float(np.abs(c - gt_n)[(gt_n >= lo) & (gt_n <= hi)].mean())} for lo, hi in CROWD_BINS]
    return s, c, fs


def show(tag, s):
    p = s["parts"]
    print(f"{tag:22s} 머리 {s['head_conf']:.2f} 전신 {s['body_conf']}  MAE {s['MAE']:.2f}  찾은 비율 {s['found_ratio']:.1%}  재현율 {s['recall']:.1%}  "
          f"정밀도 {s['precision']:.1%}  짝 {p['pair']:,} 머리만 {p['head_only']:,} 전신만 {p['body_only']:,}", flush=True)


def crowdhuman(limit=None):
    VAL = ROOT / "data" / "head_yolo"
    imgs = sorted((VAL / "images" / "val").glob("*.jpg"))[:limit]
    heads = detect(YOLO(str(ROOT / "models/head_base.pt")), imgs, 960, 0.05)
    bodies = detect(YOLO(str(ROOT / "models/yolo11n.pt")), imgs, 960, 0.2, [0])
    gts = []
    for p in imgs:
        H, W = imread(p).shape[:2]; gts.append(load_gt(VAL / "labels" / "val" / (p.stem + ".txt"), W, H))
    gt_n = np.array([len(g) for g in gts]); none = [(np.zeros((0, 4)), np.zeros(0))] * len(imgs)
    res = {"images": len(imgs), "modes": {}}
    # 전신만: 머리를 하나도 안 쓰면 전신만 박스 = 전신 개수 (기준 0.25 = Ultralytics 기본)
    res["modes"]["body"], _, _ = stats(none, bodies, gts, 0.99, 0.25); show("전신만", res["modes"]["body"])
    th = calibrate(heads, none, None, gt_n)
    res["modes"]["head"], _, _ = stats(heads, none, gts, th, None); show("머리만", res["modes"]["head"])
    for tb in BCONFS:
        th = calibrate(heads, bodies, tb, gt_n)
        res["modes"][f"fuse_b{tb}"], _, _ = stats(heads, bodies, gts, th, tb); show(f"결합(전신 {tb})", res["modes"][f"fuse_b{tb}"])
    best = min(BCONFS, key=lambda tb: res["modes"][f"fuse_b{tb}"]["MAE"])
    res["best_body_conf"] = best; res["modes"]["fuse"] = res["modes"][f"fuse_b{best}"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crowdhuman.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("가장 좋은 전신 기준:", best)


def ht21():
    from crowd_alarm import Floor, Alarm, Smoother, SMOOTH, set_levels
    from eval_video import DEMO, FOOT
    ch = json.loads((OUT / "crowdhuman.json").read_text(encoding="utf-8"))
    tb = ch["best_body_conf"]
    seq = ROOT / "data/ht21/HT21/train/HT21-03"; fps = 25.0
    imgs = sorted((seq / "img1").glob("*.jpg"))
    gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=","); gt = gt[gt[:, 7] <= 3]
    gts = []
    for p in imgs:
        g = gt[gt[:, 0] == int(p.stem)][:, 2:6]; gts.append(np.c_[g[:, :2], g[:, :2] + g[:, 2:]])
    cfg = json.loads((seq / "img1.floor.json").read_text(encoding="utf-8")); floor = Floor(cfg["floor"], cfg["size"])
    val = sorted((ROOT / "data/ht21_yolo/images/val").glob("*.jpg"))
    val_n = np.array([len([l for l in (ROOT / "data/ht21_yolo/labels/val" / (p.stem + ".txt")).read_text().splitlines() if l.strip()]) for p in val])
    body = YOLO(str(ROOT / "models/yolo11n.pt"))
    bodies = detect(body, imgs, 1920, 0.2, [0]); bodies_val = detect(body, val, 1920, 0.2, [0])   # 전신은 입력 1920이 더 잘 찾아서 유리한 쪽으로
    none = [(np.zeros((0, 4)), np.zeros(0))] * len(imgs); none_val = none[:len(val)]

    def feet(f):
        """바닥 위 점: 머리가 있으면 머리 아래 머리 높이×FOOT, 전신만이면 박스 아래 가운데"""
        H, _, B, pairs, ho, bo = f
        a = np.stack([(H[:, 0] + H[:, 2]) / 2, H[:, 3] + FOOT * (H[:, 3] - H[:, 1])], 1) if len(H) else np.zeros((0, 2))
        b = np.stack([(B[bo, 0] + B[bo, 2]) / 2, B[bo, 3]], 1) if len(bo) else np.zeros((0, 2))
        return np.concatenate([a, b])

    def density(points_per_frame):
        sm = Smoother(SMOOTH * fps); roi, maxd = [], []
        for pts in points_per_frame:
            cells, n_in = floor.density(pts); roi.append(n_in); maxd.append(float(sm([c[0] for c in cells]).max()))
        return np.array(roi), np.array(maxd)

    def alarms(maxd, roi):
        set_levels(DEMO); al = Alarm(); [al.update(i / fps, d, n) for i, (d, n) in enumerate(zip(maxd, roi))]
        set_levels((3, 4, 5)); return al.rows

    gc = np.array([len(g) for g in gts])
    groi, gmax = density([np.stack([(g[:, 0] + g[:, 2]) / 2, g[:, 3] + FOOT * (g[:, 3] - g[:, 1])], 1) for g in gts])
    res = {"seq": "HT21-03", "frames": len(imgs), "body_conf": tb, "body_imgsz": 1920, "gt_alarms_시연": alarms(gmax, groi), "models": {}}
    for name in ("head_base", "head_ht21"):
        m = YOLO(str(ROOT / "models" / f"{name}.pt")); heads = detect(m, imgs, 960, 0.05); res["models"][name] = {}
        if name == "head_ht21":                      # 기준은 HT21-02 검증에서 (시험 영상 03은 안 씀)
            hv = detect(m, val, 960, 0.05)
            th_h = calibrate(hv, none_val, None, val_n); th_f = calibrate(hv, bodies_val, tb, val_n)
        else:                                        # CrowdHuman에서 정한 값 그대로
            th_h = ch["modes"]["head"]["head_conf"]; th_f = ch["modes"]["fuse"]["head_conf"]
        for mode, (th, b_, tb_) in {"head": (th_h, none, None), "fuse": (th_f, bodies, tb)}.items():
            s, c, fs = stats(heads, b_, gts, th, tb_, crowd=False)
            roi, maxd = density([feet(f) for f in fs])
            s.update({"count_MAE": s.pop("MAE"), "count_corr": float(np.corrcoef(c, gc)[0, 1]), "maxd_ratio": float(maxd.mean() / gmax.mean()),
                      "maxd_corr": float(np.corrcoef(maxd, gmax)[0, 1]), "alarms_시연": alarms(maxd, roi)})
            res["models"][name][mode] = s
            print(f"{name:10s} {mode:5s} 머리 {th:.2f} 전신 {tb_}  찾은 비율 {s['found_ratio']:.1%}  인원 오차 {s['count_MAE']:.0f}  재현율 {s['recall']:.1%}  정밀도 {s['precision']:.1%}  "
                  f"밀도 비 {s['maxd_ratio']:.2f}  흐름 일치 {s['maxd_corr']:.2f}  부분 {s['parts']}  경보 {[(a[0], a[1]) for a in s['alarms_시연']]}", flush=True)
    (OUT / "ht21.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item()), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "crowdhuman":
        crowdhuman(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        ht21()
