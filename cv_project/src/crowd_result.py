"""Concept result image: real YOLO11n person boxes + real DM-Count density summed per ground cell.
Ground size is an assumption (perspective grid), labelled as such."""
# --- 경로 설정 (cv_project 폴더 기준, Windows/Linux 공용) ---
import os as _os, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parent.parent
REFS = str(ROOT / "refs")          # git clone 한 참조 저장소(DM-Count, CrowdCounting-P2PNet)
MODELS = str(ROOT / "models")      # yolo11n.pt 등
RESULTS = str(ROOT / "results")
def _font(bold=True):
    for p in (["C:/Windows/Fonts/malgunbd.ttf"] if bold else []) + ["C:/Windows/Fonts/malgun.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-%s.ttc" % ("Bold" if bold else "Regular")]:
        if _os.path.exists(p): return p
    raise FileNotFoundError("한글 글꼴(맑은 고딕)을 찾지 못했습니다")
FONT_B, FONT_R = _font(True), _font(False)
# -------------------------------------------------------------
import sys, os
import numpy as np, cv2, torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
sys.path.insert(0, REFS + "/DM-Count")
import torch.utils.model_zoo as mz; mz.load_url = lambda *a, **k: {}
from models import vgg19
from ultralytics import YOLO

src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
W, H = im.size
img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# X: YOLO person detection (real run)
yolo = YOLO(MODELS + "/yolo11n.pt")
r = yolo.predict(src, classes=[0], conf=0.25, imgsz=1920, verbose=False)[0]
boxes = r.boxes.xyxy.numpy()
print("yolo persons", len(boxes))

# head-based count stand-in: DM-Count density map (real run, QNRF weights)
dm = vgg19(); dm.load_state_dict(torch.load(REFS + "/DM-Count/pretrained_models/model_qnrf.pth", map_location="cpu")); dm.eval()
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
with torch.no_grad():
    den, _ = dm(tf(im).unsqueeze(0))
den = den[0, 0].numpy()
total = den.sum()
den_full = cv2.resize(den, (W, H), interpolation=cv2.INTER_LINEAR)
den_full *= total / den_full.sum()
print("dm total", total)

# Y2: ground scale from people's height (YOLO boxes, ~1.65 m) and an assumed 60 deg horizontal FOV
b = boxes; hp_b = b[:, 3] - b[:, 1]; ar = hp_b / (b[:, 2] - b[:, 0])
k = (b[:, 3] < H - 5) & (ar > 1.8)
sl, ic = np.linalg.lstsq(np.vstack([b[k, 3], np.ones(k.sum())]).T, hp_b[k], rcond=None)[0]
yhz = -ic / sl
PH, f = 1.65, (W / 2) / np.tan(np.radians(30))
print("fit slope", sl, "horizon", yhz, "people used", k.sum())
hp = lambda y: sl * (y - yhz)
# head pixels -> foot row -> ground (X, Z) metres
ys, xs = np.mgrid[0:H, 0:W]
valid = ys > yhz + 5
yf = (ys - 0.9 * sl * yhz) / (1 - 0.9 * sl)
Z = f * PH / np.maximum(hp(yf), 1e-3)
X = (xs - W / 2) * PH / np.maximum(hp(yf), 1e-3)
CELL, ZMIN, ZMAX = 2.0, 2.0, float(os.environ.get('ZMAX', 24))
sel = valid & (Z >= ZMIN) & (Z < ZMAX)
ci = np.floor(X[sel] / CELL).astype(int); cj = np.floor((Z[sel] - ZMIN) / CELL).astype(int)
acc = {}
for i, j, v in zip(ci, cj, den_full[sel]):
    acc[(i, j)] = acc.get((i, j), 0.0) + v
def to_img(Xm, Zm):
    y = yhz + (f * PH / Zm) / sl
    return W / 2 + Xm * hp(y) / PH, y
overlay = img.copy()
cells = []
for (i, j), n in acc.items():
    x0, z0 = i * CELL, ZMIN + j * CELL
    pts = np.float32([to_img(x0, z0), to_img(x0 + CELL, z0), to_img(x0 + CELL, z0 + CELL), to_img(x0, z0 + CELL)])
    cx, cy = pts.mean(axis=0)
    if cx < 0 or cx > W or pts[:, 1].max() > H + 2 or pts[:, 0].min() < -2 or pts[:, 0].max() > W + 2:
        continue  # only cells fully inside the frame
    cells.append((pts, n / (CELL * CELL), n))

def level(d):
    if d >= 5: return (40, 40, 220), "심각"
    if d >= 4: return (0, 140, 255), "경계"
    if d >= 3: return (0, 215, 255), "주의"
    if d >= 2: return (60, 200, 60), "정상"
    if d >= 1: return (150, 230, 150), "정상"
    return None, "정상"

for p, d, n in cells:
    col, _ = level(d)
    if col:
        cv2.fillConvexPoly(overlay, p.astype(np.int32), col)
img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
for p, d, n in cells:
    cv2.polylines(img, [p.astype(np.int32)], True, (235, 235, 235), 1, cv2.LINE_AA)
for x1, y1, x2, y2 in boxes.astype(int):
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 160, 0), 1)

# labels on the densest cells
pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
dr = ImageDraw.Draw(pil)
F = lambda s: ImageFont.truetype(FONT_B, s)
top = sorted(cells, key=lambda c: -c[1])
tr = lambda d: np.floor(d * 10) / 10
for p, d, n in [c for c in top if 110 < c[0].mean(axis=0)[1] < H - 70][:3]:
    cx, cy = p.mean(axis=0)
    t = f"{tr(d):.1f}명/㎡"
    bw = dr.textlength(t, font=F(26))
    dr.rounded_rectangle([cx-bw/2-6, cy-18, cx+bw/2+6, cy+18], 6, fill=(20, 20, 20))
    dr.text((cx-bw/2, cy-17), t, font=F(26), fill=(255, 255, 255))
maxd = top[0][1]
_, lv = level(maxd)
bar = 64
dr.rectangle([0, 0, W, bar + 34], fill=(17, 30, 50))
dr.text((20, 10), f"사람 {len(boxes)}명 탐지(X) · 추정 인원 {total:.0f}명 · 최대 ㎡당 {tr(maxd):.1f}명", font=F(32), fill=(255, 255, 255))
dr.text((22, bar - 4), "개념 시연: 인원은 DM-Count 밀도 지도로 대신 셈(머리 탐지 학습 전), 바닥 넓이는 사람 키 1.65m로 추정, 2m 칸", font=F(20), fill=(190, 200, 215))
badge = {"심각": (220, 40, 40), "경계": (255, 140, 0), "주의": (255, 215, 0), "정상": (60, 170, 90)}[lv]
bt = f"경보: {lv}"
bw = dr.textlength(bt, font=F(34))
dr.rounded_rectangle([W-bw-44, 8, W-14, bar-8], 10, fill=badge)
dr.text((W-bw-29, 10), bt, font=F(34), fill=(20, 20, 20) if lv in ("주의",) else (255, 255, 255))
# legend
lg = [("1~2명", (150, 230, 150)), ("2~3명", (60, 200, 60)), ("주의 3명↑", (255, 215, 0)), ("경계 4명↑", (255, 140, 0)), ("심각 5명↑", (220, 40, 40))]
x0, y0 = 20, H - 50
dr.rounded_rectangle([x0-10, y0-10, x0+5*185-20, y0+40], 8, fill=(17, 30, 50))
for i, (t, c) in enumerate(lg):
    dr.rectangle([x0 + i*185, y0+4, x0 + i*185 + 24, y0+28], fill=c)
    dr.text((x0 + i*185 + 32, y0-2), t, font=F(26), fill=(255, 255, 255))
pil.save(out, quality=92)
print("max", maxd, lv, "cells", len(cells), "levels", sorted(set(level(c[1])[1] for c in cells)))
