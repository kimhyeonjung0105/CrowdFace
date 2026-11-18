"""바닥 칸 밀도·경보 비교 (HT21-03): 머리 탐지만 vs 밀도 지도 + 머리 크기 직선으로 발 위치 vs 1판 결합.

usage: python src/eval_floor_hybrid.py         (2프레임마다 1장, 500장)
왜: 사진 단위 인원(eval_hybrid.py)보다 우리 시스템에 중요한 것은 '바닥 2m 칸의 ㎡당 밀도와 경보'.
밀도 지도는 '화면 위치마다 몇 명'만 알고 발이 바닥 어디인지 모른다 →
  우리 머리 탐지로 '화면 높이 y에서 머리 높이 h(y)' 직선을 맞추고(hybrid_count.head_size_line),
  밀도 지도 칸마다 머리 중심 = 칸 중심, 발 = y + h/2 + 6h 로 놓아 바닥으로 옮긴다(crowd_alarm.py와 같은 발 규칙).
"""
import sys, json, pathlib, numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT / "src"))
from cvio import imread
import hybrid_count as H
from crowd_alarm import Floor, Smoother, Alarm, set_levels, level_of
from ultralytics import YOLO
seq = ROOT / "data/ht21/HT21/train/HT21-03"; imgs = sorted((seq / "img1").glob("*.jpg"))[::2]; fps = 12.5
gt = np.loadtxt(seq / "gt/gt.txt", delimiter=","); gt = gt[gt[:, 7] <= 3]
cfg = json.loads((seq / "img1.floor.json").read_text(encoding="utf-8")); fl = Floor(cfg["floor"], cfg["size"])
DM = H.DensityMap(); m = YOLO(str(ROOT / "models/head_ht21.pt"))
# 발 배수: 정답 박스와 모델 박스 각각 전신 YOLO로 잰다 (foot_calib.py, 2026-11-18: 예전 고정값 6은 1.5m 어긋났음)
from foot_calib import calibrate_foot
_cal = sorted((seq / "img1").glob("*.jpg"))[::25]; _calimg = [imread(p) for p in _cal]
_g = iter([(lambda g: np.c_[g[:, :2], g[:, :2] + g[:, 2:]])(gt[gt[:, 0] == int(p.stem)][:, 2:6]) for p in _cal])
K_GT = calibrate_foot(_calimg, lambda im: next(_g))[0]
K_DET = calibrate_foot(_calimg, lambda im: m.predict(im, imgsz=960, conf=0.20, max_det=3000, verbose=False)[0].boxes.xyxy.cpu().numpy())[0]
print(f"발 배수: 정답 박스 ×{K_GT:.2f}, 모델 박스 ×{K_DET:.2f}")

def cells_from(points, weights):
    """바닥 점(가중치) → 칸별 ㎡당 인원"""
    g = fl.to_floor(points); ok = (g[:, 0] >= 0) & (g[:, 0] < fl.w) & (g[:, 1] >= 0) & (g[:, 1] < fl.d)
    cnt = {}
    for (x, y), wt in zip(g[ok], weights[ok]):
        k = (int(x // 2), int(y // 2)); cnt[k] = cnt.get(k, 0) + wt
    return np.array([cnt.get((i, j), 0) / a for i, j, a, _ in fl.cells]), float(weights[ok].sum())

def feet_boxes(b, k=None): return np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3] + (K_DET if k is None else k) * (b[:, 3] - b[:, 1])]

def dm_points(d, sc, line_h):
    """밀도 지도 칸마다: 머리 중심 = 칸 중심, 머리 높이 = 직선 h(y) → 발 = y + h/2 + 6h"""
    ii, jj = np.nonzero(d > 1e-4); x = (jj + 0.5) * sc[0]; y = (ii + 0.5) * sc[1]; h = line_h(y)
    return np.c_[x, y + 0.5 * h + K_DET * h], d[ii, jj]

R = {k: {"maxd": [], "roi": []} for k in ("gt", "det", "dm", "hyb1")}; S = {k: Smoother(fps) for k in R}
for p in imgs:
    img = imread(p); g = gt[gt[:, 0] == int(p.stem)][:, 2:6]; gb = np.c_[g[:, :2], g[:, :2] + g[:, 2:]]
    b = m.predict(img, imgsz=960, conf=0.20, max_det=3000, verbose=False)[0].boxes.xyxy.cpu().numpy()
    d, sc = DM(img)
    hb = b.copy(); hb[:, 2] = hb[:, 0] + (b[:, 3] - b[:, 1])          # 높이 직선을 맞추려고 폭 자리에 높이를 넣음
    line_h = H.head_size_line(hb)
    out = {}
    out["gt"] = cells_from(feet_boxes(gb, K_GT), np.ones(len(gb)))
    out["det"] = cells_from(feet_boxes(b), np.ones(len(b)))
    out["dm"] = cells_from(*dm_points(d, sc, line_h))
    _, cl = H.hybrid_count(b, d, sc, return_map=True); use = np.kron(cl[..., 2], np.ones((8, 8)))[:d.shape[0], :d.shape[1]].astype(bool)
    pd, wd = dm_points(np.where(use, d, 0), sc, line_h)
    bc = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]; keep = ~use[np.clip((bc[:, 1] // sc[1]).astype(int), 0, d.shape[0] - 1), np.clip((bc[:, 0] // sc[0]).astype(int), 0, d.shape[1] - 1)]
    fb = feet_boxes(b[keep]); out["hyb1"] = cells_from(np.r_[fb, pd], np.r_[np.ones(len(fb)), wd])
    for k, (c, n) in out.items():
        R[k]["maxd"].append(float(S[k](list(c)).max())); R[k]["roi"].append(n)
G = {k: np.array(v["maxd"]) for k, v in R.items()}; N = {k: np.array(v["roi"]) for k, v in R.items()}
set_levels((1.0, 1.33, 1.67))
def alarms(md, roi):
    al = Alarm(); [al.update(i / fps, x, n) for i, (x, n) in enumerate(zip(md, roi))]; return [(a[0], a[1]) for a in al.rows]
for k in ("gt", "det", "dm", "hyb1"):
    s = f"{k:5s} 구역 인원 비 {N[k].sum() / N['gt'].sum():.2f}  최대 칸 밀도 평균 {G[k].mean():.2f} (정답 {G['gt'].mean():.2f})"
    if k != "gt":
        s += f"  흐름 일치 {np.corrcoef(G[k], G['gt'])[0, 1]:.2f}  단계 일치(시연) {np.mean([level_of(a)[1] == level_of(b)[1] for a, b in zip(G[k], G['gt'])]):.0%}"
    print(s, "경보", alarms(G[k], N[k]))
