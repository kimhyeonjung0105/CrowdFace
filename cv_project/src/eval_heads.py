"""전신(X) vs 머리(Y①) 인원 세기 평가 — CrowdHuman 검증 4,370장.

usage: python src/eval_heads.py [최대장수]     (신뢰도 기준 바꾸기: EVAL_CONF=0.15 → results/eval_conf015/)
비교 모델
  body      : COCO yolo11n.pt, 사람(0번) 클래스 전신 박스 개수 = 인원   (기존 기능 X)
  head_base : 우리가 학습한 머리 모델(원본 구조)   models/head_base.pt
  head_p2   : 우리가 학습한 머리 모델(P2 구조)     models/head_p2.pt
정답 인원 = 그 사진의 머리 라벨 개수 (convert_crowdhuman.py가 만든 val 라벨, 무시 영역 제외)

출력 results/eval/
  per_image.csv  사진별 정답·모델별 인원
  summary.json   평균 절대 오차(MAE), 붐빈 정도별 MAE, 머리 크기별 재현율, 속도
"""
import os, sys, json, csv, time, pathlib as _pl
import numpy as np, torch
ROOT = _pl.Path(__file__).resolve().parent.parent
VAL = ROOT / "data" / "head_yolo"
# EVAL_CONF: 신뢰도 기준. 기본 0.25(Ultralytics 기본값, mAP용). 인원 세기용 0.15는 calib_conf.py로 검증 데이터에서 정함
CONF = float(os.environ.get("EVAL_CONF", 0.25))
OUT = ROOT / "results" / ("eval" if CONF == 0.25 else f"eval_conf{int(round(CONF * 100)):03d}")
MODELS = _pl.Path(os.environ.get("EVAL_MODELS", ROOT / "models"))   # 시험용으로 다른 폴더 지정 가능
from ultralytics import YOLO

IMGSZ = 960                        # 학습과 같은 입력 크기
CROWD_BINS = [(0, 10), (11, 30), (31, 60), (61, 10**9)]          # 사진 한 장의 정답 인원
SIZE_BINS = [(0, 16), (16, 32), (32, 64), (64, 10**9)]           # 머리 박스 높이(픽셀)
DEV = 0 if torch.cuda.is_available() else "cpu"


def load_gt(lbl, W, H):
    """YOLO 라벨(0~1 비율) → 픽셀 박스 [x1,y1,x2,y2]"""
    rows = [list(map(float, l.split()[1:])) for l in lbl.read_text().splitlines() if l.strip()]
    b = np.array(rows).reshape(-1, 4)
    return np.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H,
                     (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H], 1)


def iou(a, b):
    """a: (N,4), b: (M,4) → (N,M)"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def matched(gt, pred, conf, thr=0.5):
    """신뢰도 높은 예측부터 IoU≥0.5인 정답과 1:1로 짝지음 → 정답별 찾음 여부"""
    hit = np.zeros(len(gt), bool)
    if len(gt) == 0 or len(pred) == 0:
        return hit
    M = iou(gt, pred)
    for j in np.argsort(-conf):
        cand = np.where(~hit & (M[:, j] >= thr))[0]
        if len(cand):
            hit[cand[np.argmax(M[cand, j])]] = True
    return hit


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    imgs = sorted((VAL / "images" / "val").glob("*.jpg"))[:limit]
    models = {"body": MODELS / "yolo11n.pt"}
    for k in ("head_base", "head_p2"):
        if (MODELS / f"{k}.pt").exists():
            models[k] = MODELS / f"{k}.pt"
    OUT.mkdir(parents=True, exist_ok=True)

    gt_n, counts, speed = [], {k: [] for k in models}, {}
    size_hit = {k: [[0, 0] for _ in SIZE_BINS] for k in models if k != "body"}   # [찾음, 전체]
    gts = {}
    for k, w in models.items():
        m = YOLO(str(w)); ms = []
        cls = [0] if k == "body" else None          # 전신 모델은 COCO 80종 중 사람만
        for p in imgs:
            r = m.predict(str(p), imgsz=IMGSZ, conf=CONF, classes=cls, device=DEV, verbose=False)[0]
            ms.append(r.speed["inference"])
            counts[k].append(len(r.boxes))
            if k == "body":
                continue
            if p.name not in gts:
                H, W = r.orig_shape
                gts[p.name] = load_gt(VAL / "labels" / "val" / (p.stem + ".txt"), W, H)
            g = gts[p.name]
            hit = matched(g, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
            h = g[:, 3] - g[:, 1]
            for i, (lo, hi) in enumerate(SIZE_BINS):
                sel = (h >= lo) & (h < hi)
                size_hit[k][i][0] += int(hit[sel].sum()); size_hit[k][i][1] += int(sel.sum())
        speed[k] = float(np.mean(ms[5:] or ms))      # 처음 몇 장은 GPU 예열이라 뺌
        print(f"{k}: 완료 ({speed[k]:.1f} ms/장)")

    gt_n = [len((VAL / "labels" / "val" / (p.stem + ".txt")).read_text().split("\n"))
            if (VAL / "labels" / "val" / (p.stem + ".txt")).read_text().strip() else 0 for p in imgs]
    gt = np.array(gt_n)

    summary = {"images": len(imgs), "imgsz": IMGSZ, "conf": CONF, "gt_total": int(gt.sum()),
               "speed_ms": speed, "models": list(models)}
    for k in models:
        c = np.array(counts[k])
        summary[k] = {
            "total": int(c.sum()),
            "MAE": float(np.abs(c - gt).mean()),
            "bias": float((c - gt).mean()),                       # 음수 = 덜 셈
            "found_ratio": float(c.sum() / max(gt.sum(), 1)),
            "by_crowd": [{"range": f"{lo}-{hi if hi < 10**9 else ''}",
                          "n_img": int(((gt >= lo) & (gt <= hi)).sum()),
                          "MAE": float(np.abs(c - gt)[(gt >= lo) & (gt <= hi)].mean()) if ((gt >= lo) & (gt <= hi)).any() else None,
                          "ratio": float(c[(gt >= lo) & (gt <= hi)].sum() / max(gt[(gt >= lo) & (gt <= hi)].sum(), 1))}
                         for lo, hi in CROWD_BINS]}
        if k in size_hit:
            summary[k]["recall_by_size"] = [{"range": f"{lo}-{hi if hi < 10**9 else ''}px", "found": f, "total": t,
                                             "recall": f / t if t else None}
                                            for (lo, hi), (f, t) in zip(SIZE_BINS, size_hit[k])]
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(OUT / "per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["image", "gt"] + list(models))
        for i, p in enumerate(imgs):
            w.writerow([p.name, gt[i]] + [counts[k][i] for k in models])
    for k in models:
        print(k, "MAE %.2f  찾은 비율 %.1f%%" % (summary[k]["MAE"], 100 * summary[k]["found_ratio"]))


if __name__ == "__main__":
    main()
