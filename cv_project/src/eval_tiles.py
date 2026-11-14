"""타일링 평가: 보통 탐지(전체 사진 1번) vs 타일링 2×2 · 3×3 — 작은 머리를 더 찾는지, 인원이 더 정확한지.

usage: python src/eval_tiles.py crowdhuman [최대장수]   CrowdHuman 검증 4,370장 (head_base)
       python src/eval_tiles.py ht21                     HT21-03 영상 1,000프레임 (head_base, head_ht21)  ← crowdhuman 먼저
출력 results/eval_tiles/crowdhuman.json, ht21.json

공정하게 비교하는 법
  타일링은 박스를 더 많이 내므로 같은 신뢰도 기준으로 세면 당연히 더 많이 센다(헛것 포함).
  그래서 방식마다 '검증 데이터에서 센 인원 합 = 실제 인원 합'이 되는 기준을 따로 정하고(calib_conf.py와 같은 규칙),
  **똑같이 100%를 셀 때 진짜 머리를 얼마나 맞혔는지(재현율)** 를 비교한다. 같은 개수를 세는데 재현율이 높으면 헛것이 적다는 뜻.
  - CrowdHuman: 검증 사진으로 기준을 정하고 같은 사진으로 평가(약간 유리한 값, 보고서 4-4와 같은 조건)
  - HT21: head_ht21의 기준은 HT21-02 뒤 20%(검증)에서 정하고, 시험 영상 HT21-03에는 손대지 않음.
          head_base의 기준은 CrowdHuman에서 정한 값 그대로.
"""
import sys, json, time, pathlib as _pl
import numpy as np
ROOT = _pl.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cvio import imread
from tile_detect import detect_tiled
from eval_heads import load_gt, matched, SIZE_BINS, CROWD_BINS
from ultralytics import YOLO
OUT = ROOT / "results" / "eval_tiles"
MODES = {"full": 1, "tile2": 2, "tile3": 3}        # 이름: 격자 n (1 = 보통 탐지)
LOW = 0.05                                          # 이 값 이상 박스를 모두 모아 두고, 기준은 나중에 바꿔 가며 셈
CONFS = np.round(np.arange(0.05, 0.61, 0.01), 2)


def run(model, imgs, n):
    """사진마다 (박스, 신뢰도) 목록과 장당 걸린 시간(ms)"""
    dets, t0 = [], None
    for i, p in enumerate(imgs):
        if i == 5:
            t0 = time.perf_counter()                # 처음 몇 장은 GPU 예열이라 시간에서 뺌
        dets.append(detect_tiled(model, imread(p), n, conf=LOW))
    ms = (time.perf_counter() - t0) / max(len(imgs) - 5, 1) * 1000 if t0 else 0.0
    return dets, ms


def calibrate(dets, gt_n):
    """센 인원 합이 실제 인원 합과 가장 가까운 신뢰도 기준"""
    ratio = [sum(int((c >= t).sum()) for _, c in dets) / gt_n.sum() for t in CONFS]
    return float(CONFS[int(np.argmin(np.abs(np.array(ratio) - 1)))])


def count_stats(dets, gt_n, t):
    c = np.array([int((cf >= t).sum()) for _, cf in dets])
    return c, {"conf": t, "MAE": float(np.abs(c - gt_n).mean()), "found_ratio": float(c.sum() / gt_n.sum())}


def crowdhuman(limit=None):
    VAL = ROOT / "data" / "head_yolo"
    imgs = sorted((VAL / "images" / "val").glob("*.jpg"))[:limit]
    model = YOLO(str(ROOT / "models" / "head_base.pt"))
    gts = None; res = {"images": len(imgs), "model": "head_base", "modes": {}}
    for mode, n in MODES.items():
        dets, ms = run(model, imgs, n)
        if gts is None:                              # 정답 박스(픽셀)는 사진 크기가 필요해서 첫 방식 때 한 번 읽음
            gts = []
            for p in imgs:
                H, W = imread(p).shape[:2]
                gts.append(load_gt(VAL / "labels" / "val" / (p.stem + ".txt"), W, H))
            gt_n = np.array([len(g) for g in gts])
        t = calibrate(dets, gt_n)
        r = {"ms_per_image": ms}
        for tag, tt in (("calibrated", t), ("conf015", 0.15)):
            c, s = count_stats(dets, gt_n, tt)
            s["by_crowd"] = [{"range": f"{lo}-{hi if hi < 10**9 else ''}", "ratio": float(c[(gt_n >= lo) & (gt_n <= hi)].sum() / max(gt_n[(gt_n >= lo) & (gt_n <= hi)].sum(), 1)),
                              "MAE": float(np.abs(c - gt_n)[(gt_n >= lo) & (gt_n <= hi)].mean())} for lo, hi in CROWD_BINS]
            hit_n = [[0, 0] for _ in SIZE_BINS]; tp = 0
            for g, (b, cf) in zip(gts, dets):
                k = cf >= tt
                hit = matched(g, b[k], cf[k]); tp += int(hit.sum())
                h = g[:, 3] - g[:, 1]
                for i, (lo, hi) in enumerate(SIZE_BINS):
                    sel = (h >= lo) & (h < hi)
                    hit_n[i][0] += int(hit[sel].sum()); hit_n[i][1] += int(sel.sum())
            s["recall"] = tp / gt_n.sum(); s["precision"] = tp / max(c.sum(), 1)     # 맞힌 머리 ÷ 정답 / ÷ 센 개수
            s["recall_by_size"] = [{"range": f"{lo}-{hi if hi < 10**9 else ''}px", "found": f, "total": tot, "recall": f / tot if tot else None}
                                   for (lo, hi), (f, tot) in zip(SIZE_BINS, hit_n)]
            r[tag] = s
        res["modes"][mode] = r
        s = r["calibrated"]
        print(f"{mode:6s} 기준 {s['conf']:.2f}  MAE {s['MAE']:.2f}  찾은 비율 {s['found_ratio']:.1%}  재현율 {s['recall']:.1%}  정밀도 {s['precision']:.1%}  "
              + " ".join(f"{x['range']} {x['recall']:.1%}" for x in s["recall_by_size"]) + f"  {ms:.0f} ms/장", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "crowdhuman.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def ht21():
    from crowd_alarm import Floor, Alarm, Smoother, SMOOTH, set_levels
    from eval_video import feet_from, DEMO
    ch = json.loads((OUT / "crowdhuman.json").read_text(encoding="utf-8"))["modes"]
    seq = ROOT / "data/ht21/HT21/train/HT21-03"; fps = 25.0
    imgs = sorted((seq / "img1").glob("*.jpg"))
    gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=","); gt = gt[gt[:, 7] <= 3]
    cfg = json.loads((seq / "img1.floor.json").read_text(encoding="utf-8")); floor = Floor(cfg["floor"], cfg["size"])
    val = sorted((ROOT / "data/ht21_yolo/images/val").glob("*.jpg"))                 # HT21-02 뒤 20% (기준 정하기용)
    val_n = np.array([len([l for l in (ROOT / "data/ht21_yolo/labels/val" / (p.stem + ".txt")).read_text().splitlines() if l.strip()]) for p in val])

    def series(boxes_per_frame):
        sm = Smoother(SMOOTH * fps); cnt, roi, maxd = [], [], []
        for b in boxes_per_frame:
            cells, n_in = floor.density(feet_from(b, "head"))
            cnt.append(len(b)); roi.append(n_in); maxd.append(float(sm([c[0] for c in cells]).max()))
        return np.array(cnt), np.array(roi), np.array(maxd)

    def alarms(maxd, roi):
        set_levels(DEMO); al = Alarm(); [al.update(i / fps, d, n) for i, (d, n) in enumerate(zip(maxd, roi))]
        set_levels((3, 4, 5)); return al.rows

    g_boxes = []
    for p in imgs:
        g = gt[gt[:, 0] == int(p.stem)][:, 2:6]; g_boxes.append(np.c_[g[:, :2], g[:, :2] + g[:, 2:]])
    gc, groi, gmax = series(g_boxes)
    res = {"seq": "HT21-03", "frames": len(imgs), "gt_mean_count": float(gc.mean()), "gt_alarms_시연": alarms(gmax, groi), "models": {}}
    for name in ("head_base", "head_ht21"):
        model = YOLO(str(ROOT / "models" / f"{name}.pt")); res["models"][name] = {}
        for mode, n in MODES.items():
            if name == "head_ht21":
                t = calibrate(run(model, val, n)[0], val_n); src = "HT21-02 검증"
            else:
                t = ch[mode]["calibrated"]["conf"]; src = "CrowdHuman 검증"
            dets, ms = run(model, imgs, n)
            c, roi, maxd = series([b[cf >= t] for b, cf in dets])
            r = {"conf": t, "conf_from": src, "count_ratio": float(c.sum() / gc.sum()), "count_MAE": float(np.abs(c - gc).mean()),
                 "count_corr": float(np.corrcoef(c, gc)[0, 1]), "maxd_bias": float((maxd - gmax).mean()),
                 "maxd_ratio": float(maxd.mean() / gmax.mean()), "maxd_corr": float(np.corrcoef(maxd, gmax)[0, 1]),
                 "alarms_시연": alarms(maxd, roi), "ms_per_image": ms, "count_series": c.tolist(), "maxd_series": maxd.tolist()}
            res["models"][name][mode] = r
            print(f"{name:10s} {mode:6s} 기준 {t:.2f}({src})  찾은 비율 {r['count_ratio']:.1%}  인원 오차 {r['count_MAE']:.0f}  "
                  f"밀도 비 {r['maxd_ratio']:.2f}  흐름 일치 {r['maxd_corr']:.2f}  경보 {[(a[0], a[1]) for a in r['alarms_시연']]}  {ms:.0f} ms/장", flush=True)
    res["gt_count_series"] = gc.tolist(); res["gt_maxd_series"] = gmax.tolist()
    (OUT / "ht21.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item()), encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1] == "crowdhuman":
        crowdhuman(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        ht21()
