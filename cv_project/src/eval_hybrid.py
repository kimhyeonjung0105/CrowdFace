"""결합 세기 평가: 칸 크기·빽빽함 기준을 JHU 검증 500장에서 고르고, 시험 1,600장에서 평가.

usage: python src/eval_hybrid.py            → results/eval_hybrid/summary.json  (탐지·밀도 지도는 cache_*.npz에 저장, GitHub 제외)
비교: 머리 탐지만(보통·타일링) / 밀도 지도만 / 사진 전체 하나로 고르기(인원 T명 넘으면 밀도 지도) / 칸마다 고르기(hybrid_count)
공정성: 기준값은 검증(val)에서만 고른다. 시험(test)에서 고르면 시험 답을 보고 고른 것이 된다.
"""
import sys, json, itertools, pathlib as _pl
import numpy as np
ROOT = _pl.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from hybrid_count import hybrid_count
OUT = ROOT / "results" / "eval_hybrid"
JHU = ROOT / "data" / "jhu" / "jhu_crowd_v2.0"
COUNT_BINS = [(0, 50), (51, 200), (201, 500), (501, 1000), (1001, 3000), (3001, 10**6)]
GRID = {"block": [64, 128], "c0": [1.25, 1.5, 2.0, 2.5], "m0": [2.0, 4.0, 8.0, 1e9], "r": [1.5, 2.0, 3.0, 1e9], "t_img": [0, 50, 100, 200]}


def build_cache(split):
    """사진마다 머리 박스(보통·타일링), 밀도 지도, 정답 인원 → cache_<split>.npz"""
    p = OUT / f"cache_{split}.npz"
    if p.exists():
        return dict(np.load(p, allow_pickle=True))
    from cvio import imread
    from tile_detect import detect_tiled
    from hybrid_count import DensityMap
    from ultralytics import YOLO
    import torch
    head = YOLO(str(ROOT / "models/head_base.pt")); dm = DensityMap()
    lab = [l.split(",") for l in (JHU / split / "image_labels.txt").read_text().split("\n") if l.strip()]
    full, tile, dmaps, scales, gt = [], [], [], [], []
    for i, l in enumerate(lab):
        img = imread(JHU / split / "images" / f"{l[0]}.jpg")
        full.append(detect_tiled(head, img, 1, conf=0.15, max_det=10000)[0].astype(np.float32))
        tile.append(detect_tiled(head, img, 2, conf=0.25, max_det=10000)[0].astype(np.float32))
        d, sc = dm(img); dmaps.append(d.astype(np.float32)); scales.append(sc); gt.append(int(l[1]))
        torch.cuda.empty_cache()
        if i % 200 == 0:
            print(f"{split} {i}/{len(lab)}", flush=True)
    obj = lambda L: np.array(L + [None], dtype=object)[:-1]     # 크기가 다른 배열 목록 저장
    c = {"full": obj(full), "tile": obj(tile), "dmap": obj(dmaps), "scale": np.array(scales), "gt": np.array(gt)}
    OUT.mkdir(parents=True, exist_ok=True); np.savez_compressed(p, **c)
    return c


def stats(pred, gt):
    pred = np.asarray(pred, float)
    return {"MAE": float(np.abs(pred - gt).mean()), "RMSE": float(np.sqrt(((pred - gt) ** 2).mean())), "found_ratio": float(pred.sum() / gt.sum()),
            "by_count": [{"range": f"{lo}-{hi if hi < 10**6 else ''}", "n_img": int(((gt >= lo) & (gt <= hi)).sum()),
                          "MAE": float(np.abs(pred - gt)[(gt >= lo) & (gt <= hi)].mean()),
                          "ratio": float(pred[(gt >= lo) & (gt <= hi)].sum() / max(gt[(gt >= lo) & (gt <= hi)].sum(), 1))} for lo, hi in COUNT_BINS]}


def main():
    val, test = build_cache("val"), build_cache("test")
    res = {"val_images": len(val["gt"]), "test_images": len(test["gt"]), "grid": GRID}
    best = {}
    for det in ("full", "tile"):
        # 칸마다 고르기: 검증에서 MAE가 가장 작은 조합
        scores = []
        for prm in itertools.product(*GRID.values()):
            pred = [hybrid_count(x, d, s, *prm) for x, d, s in zip(val[det], val["dmap"], val["scale"])]
            scores.append((float(np.abs(np.array(pred) - val["gt"]).mean()),) + prm)
        scores.sort(); best[det] = dict(zip(("val_MAE",) + tuple(GRID), scores[0]))
        best[det]["rules_off"] = [s for s in scores if s[4] == 1e9 and s[5] == 0][0][0]     # 규칙 0)·R 없이 고른 것의 검증 MAE
        # 사진 전체 하나로 고르기: 밀도 지도 합이 T명 넘으면 밀도 지도
        dsum_v = np.array([d.sum() for d in val["dmap"]]); nv = np.array([len(x) for x in val[det]])
        Ts = [0, 25, 50, 100, 150, 200, 300, 500, 1000, 1e9]
        tv = min(Ts, key=lambda T: np.abs(np.where(dsum_v > T, dsum_v, nv) - val["gt"]).mean())
        best[det]["image_T"] = tv
        print(f"[{det}] 칸마다: {best[det]} → 검증 MAE {scores[0][0]:.1f} (추가 규칙 없이 {best[det]['rules_off']:.1f})   사진 전체 기준 T={tv}", flush=True)
    res["best"] = best
    # 시험 평가 (고른 기준 그대로)
    dsum = np.array([d.sum() for d in test["dmap"]]); gt = test["gt"]; M = {}
    M["밀도 지도만 (DM-Count)"] = dsum
    for det, nm in (("full", "보통"), ("tile", "타일링")):
        n = np.array([len(x) for x in test[det]]); bb = best[det]
        M[f"머리 탐지만 ({nm})"] = n
        M[f"사진 전체로 고르기 ({nm})"] = np.where(dsum > bb["image_T"], dsum, n)
        M[f"칸마다 고르기 ({nm})"] = np.array([hybrid_count(x, d, s, *(bb[k] for k in GRID))
                                          for x, d, s in zip(test[det], test["dmap"], test["scale"])])
    res["test"] = {k: stats(v, gt) for k, v in M.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, s in res["test"].items():
        print(f"{k:22s} MAE {s['MAE']:7.1f}  RMSE {s['RMSE']:7.1f}  " + "  ".join(f"{b['range']} {b['MAE']:.0f}({b['ratio']:.0%})" for b in s["by_count"]))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# ---------------- 칸별 정확도 (GAME) ----------------
def raster(xy, scale, shape):
    """점들을 밀도 지도 격자 위 개수 지도로 (밀도 지도 한 칸 = 원본 scale 픽셀)"""
    m = np.zeros(shape)
    if len(xy):
        i = np.clip((xy[:, 1] // scale[1]).astype(int), 0, shape[0] - 1); j = np.clip((xy[:, 0] // scale[0]).astype(int), 0, shape[1] - 1)
        np.add.at(m, (i, j), 1)
    return m


def game(pred, gtm, L):
    """GAME(L): 사진을 2^L × 2^L 칸으로 나눠 칸마다 |예측 - 정답|을 더함 (L=0이면 보통 인원 오차). 군중 세기 논문의 표준 지역 오차"""
    h, w = gtm.shape; n = 2 ** L; e = 0.0
    ys = np.linspace(0, h, n + 1).round().astype(int); xs = np.linspace(0, w, n + 1).round().astype(int)
    for a in range(n):
        for b in range(n):
            e += abs(pred[ys[a]:ys[a + 1], xs[b]:xs[b + 1]].sum() - gtm[ys[a]:ys[a + 1], xs[b]:xs[b + 1]].sum())
    return e


def eval_game():
    test = build_cache("test"); res = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    bb = res["best"]["full"]; prm = [bb[k] for k in GRID]
    lab = [l.split(",") for l in (JHU / "test" / "image_labels.txt").read_text().split("\n") if l.strip()]
    E = {k: np.zeros(4) for k in ("머리 탐지만", "밀도 지도만", "칸마다 고르기")}; n_img = 0
    for l, boxes, d, sc in zip(lab, test["full"], test["dmap"], test["scale"]):
        g = np.loadtxt(JHU / "test" / "gt" / f"{l[0]}.txt", ndmin=2).reshape(-1, 6)
        gtm = raster(g[:, :2], sc, d.shape)
        det = raster(np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], sc, d.shape)
        _, cells = hybrid_count(boxes, d, sc, *prm, return_map=True)
        k = prm[0] // 8; use = np.kron(cells[..., 2], np.ones((k, k)))[:d.shape[0], :d.shape[1]].astype(bool)
        hyb = np.where(use, d, det)
        for L in range(4):
            E["머리 탐지만"][L] += game(det, gtm, L); E["밀도 지도만"][L] += game(d, gtm, L); E["칸마다 고르기"][L] += game(hyb, gtm, L)
        n_img += 1
    res["test_GAME"] = {k: {f"GAME{L}": float(v[L] / n_img) for L in range(4)} for k, v in E.items()}
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in res["test_GAME"].items():
        print(f"{k:10s} " + "  ".join(f"{n} {x:.1f}" for n, x in v.items()))


GRID2 = {"c0": [1.0, 1.25, 1.5, 2.0], "w_small": [4, 6, 8, 10, 12], "w_big": [12, 16, 24, 32, 1e9], "q": [0.05, 0.1, 0.2, 0.3]}


def eval_v2():
    """2판(원근으로 머리 크기 예측): 검증에서 기준 고르기 → 시험 인원 오차·GAME"""
    from hybrid_count import hybrid_count2
    val, test = build_cache("val"), build_cache("test"); res = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    scores = []
    for prm in itertools.product(*GRID2.values()):
        pred = [hybrid_count2(x, d, s, *prm) for x, d, s in zip(val["full"], val["dmap"], val["scale"])]
        scores.append((float(np.abs(np.array(pred) - val["gt"]).mean()),) + prm)
    scores.sort(); best = dict(zip(("val_MAE",) + tuple(GRID2), scores[0]))
    best["no_fake_rule"] = min(s for s in scores if s[3] == 1e9)[0]               # 헛것 규칙 ② 없이
    print("2판 검증:", best, flush=True)
    prm = [best[k] for k in GRID2]
    pred = np.array([hybrid_count2(x, d, s, *prm) for x, d, s in zip(test["full"], test["dmap"], test["scale"])])
    res["best_v2"] = best; res["test"]["칸마다 고르기 2판 (원근)"] = stats(pred, test["gt"])
    lab = [l.split(",") for l in (JHU / "test" / "image_labels.txt").read_text().split("\n") if l.strip()]
    E = np.zeros(4)
    for l, boxes, d, sc in zip(lab, test["full"], test["dmap"], test["scale"]):
        g = np.loadtxt(JHU / "test" / "gt" / f"{l[0]}.txt", ndmin=2).reshape(-1, 6); gtm = raster(g[:, :2], sc, d.shape)
        det = raster(np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], sc, d.shape)
        _, cells = hybrid_count2(boxes, d, sc, *prm, return_map=True)
        k = 64 // 8                                            # 2판·1판 모두 블록 64px = 밀도 지도 8칸
        use = np.kron(cells[..., 2], np.ones((k, k)))[:d.shape[0], :d.shape[1]].astype(bool)
        hyb = np.where(use, d, det)
        for L in range(4):
            E[L] += game(hyb, gtm, L)
    res.setdefault("test_GAME", {})["칸마다 고르기 2판 (원근)"] = {f"GAME{L}": float(E[L] / len(lab)) for L in range(4)}
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    s = res["test"]["칸마다 고르기 2판 (원근)"]
    print(f"2판 시험 MAE {s['MAE']:.1f}  " + "  ".join(f"{b['range']} {b['MAE']:.0f}({b['ratio']:.0%})" for b in s["by_count"]))
    print("2판 GAME", res["test_GAME"]["칸마다 고르기 2판 (원근)"])


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "game":
    eval_game()
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "v2":
    eval_v2()
