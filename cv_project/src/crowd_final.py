"""Final-result style screen: head boxes (P2PNet) + density map (DM-Count) + densest 2 m cell + alert panel.
usage: python3 crowd_final.py in.jpg out.jpg SLOPE HORIZON   (person-height fit from crowd_result.py)"""
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
import sys, types
import numpy as np, cv2, torch, torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

src, out, sl, yhz = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
im = Image.open(src).convert("RGB")
W, H = im.size
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# density: DM-Count, QNRF weights
sys.path.insert(0, REFS + "/DM-Count")
import torch.utils.model_zoo as mz; mz.load_url = lambda *a, **k: {}
from models import vgg19
dm = vgg19(); dm.load_state_dict(torch.load(REFS + "/DM-Count/pretrained_models/model_qnrf.pth", map_location="cpu")); dm.eval()
with torch.no_grad():
    den, _ = dm(norm(T.ToTensor()(im)).unsqueeze(0))  # same normalisation as DM-Count training
den = den[0, 0].numpy(); total = float(den.sum())
den_full = cv2.resize(den, (W, H), interpolation=cv2.INTER_LINEAR); den_full *= total / den_full.sum()
sys.path.pop(0)
for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
    del sys.modules[k]

# heads: P2PNet, ShanghaiTech A weights
torchvision.__version__ = "0.99"
sp = types.ModuleType("scipy"); spo = types.ModuleType("scipy.optimize"); spo.linear_sum_assignment = None; sp.optimize = spo
sys.modules.update({"scipy": sp, "scipy.optimize": spo})
sys.path.insert(0, REFS + "/CrowdCounting-P2PNet")
import models.vgg_ as V
_f = V.vgg16_bn; V.vgg16_bn = lambda pretrained=False, **k: _f(pretrained=False, **k)
from models import build_model
class A: backbone = "vgg16_bn"; row = 2; line = 2
p2p = build_model(A()); p2p.load_state_dict(torch.load(REFS + "/CrowdCounting-P2PNet/weights/SHTechA.pth", map_location="cpu")["model"]); p2p.eval()
w2, h2 = W // 128 * 128, H // 128 * 128
with torch.no_grad():
    o = p2p(norm(T.ToTensor()(im.resize((w2, h2), Image.LANCZOS))).unsqueeze(0))
sc = torch.softmax(o["pred_logits"], -1)[0, :, 1]
heads = o["pred_points"][0][sc > 0.5].numpy() * [W / w2, H / h2]
print("heads", len(heads), "dm", total)

# ground cells (2 m) from the person-height scale, densest fully visible cell
PH, f = 1.65, (W / 2) / np.tan(np.radians(30))
hp = lambda y: sl * (y - yhz)
ys, xs = np.mgrid[0:H, 0:W]
yf = (ys - 0.9 * sl * yhz) / (1 - 0.9 * sl)
Z = f * PH / np.maximum(hp(yf), 1e-3); X = (xs - W / 2) * PH / np.maximum(hp(yf), 1e-3)
CELL, ZMIN, ZMAX = 2.0, 2.0, 30.0
sel = (ys > yhz + 5) & (Z >= ZMIN) & (Z < ZMAX)
acc = {}
for i, j, v in zip(np.floor(X[sel] / CELL).astype(int), np.floor((Z[sel] - ZMIN) / CELL).astype(int), den_full[sel]):
    acc[(i, j)] = acc.get((i, j), 0.0) + v
def to_img(Xm, Zm):
    y = yhz + (f * PH / Zm) / sl
    return W / 2 + Xm * hp(y) / PH, y
# walkable area marked by the operator (stalls excluded), as in the plan's "floor area only" rule
ROI = np.int32([[0, 300], [1200, 190], [1200, 900], [480, 900], [480, 640], [240, 560], [230, 470], [0, 470]])
best = None
for (i, j), n in acc.items():
    x0, z0 = i * CELL, ZMIN + j * CELL
    pts = np.float32([to_img(x0, z0), to_img(x0 + CELL, z0), to_img(x0 + CELL, z0 + CELL), to_img(x0, z0 + CELL)])
    if pts[:, 1].max() > H + 2 or pts[:, 0].min() < -2 or pts[:, 0].max() > W + 2:
        continue
    if cv2.pointPolygonTest(ROI.reshape(-1, 1, 2), tuple(map(float, pts.mean(axis=0))), False) < 0:
        continue
    d = n / CELL ** 2
    if best is None or d > best[1]:
        best = (pts, d)
maxd = np.floor(best[1] * 10) / 10
lv, lc = ("심각", (220, 40, 40)) if maxd >= 5 else ("경계", (255, 140, 0)) if maxd >= 4 else ("주의", (240, 200, 0)) if maxd >= 3 else ("정상", (60, 170, 90))

# picture: density overlay + head boxes + densest cell
d = cv2.GaussianBlur(cv2.resize(den, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 3)
d = np.clip(d / np.percentile(d, 99.7), 0, 1)
img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
img = cv2.addWeighted(img, 0.6, cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0)
for px, py in heads:
    s = min(60, max(8, 0.23 * sl * (py - yhz) / (1 - 0.87 * sl)))
    cv2.rectangle(img, (int(px - s / 2), int(py - s / 2)), (int(px + s / 2), int(py + s / 2)), (255, 255, 255), 1, cv2.LINE_AA)
cv2.polylines(img, [ROI], True, (230, 230, 230), 1, cv2.LINE_AA)
cv2.polylines(img, [best[0].astype(np.int32)], True, (255, 255, 255), 4, cv2.LINE_AA)
cv2.polylines(img, [best[0].astype(np.int32)], True, lc[::-1], 2, cv2.LINE_AA)

PW = 400
canvas = Image.new("RGB", (W + PW, H), (17, 30, 50))
canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (0, 0))
dr = ImageDraw.Draw(canvas)
F = lambda s: ImageFont.truetype(FONT_B, s)
R = lambda s: ImageFont.truetype(FONT_R, s)
cx, cy = best[0].mean(axis=0)
t = f"㎡당 {maxd:.1f}명"
tw = dr.textlength(t, font=F(24))
ty = best[0][:, 1].min() - 40
dr.rounded_rectangle([cx - tw / 2 - 8, ty, cx + tw / 2 + 8, ty + 34], 6, fill=(17, 30, 50))
dr.text((cx - tw / 2, ty + 1), t, font=F(24), fill=(255, 255, 255))

x0 = W + 32
dr.text((x0, 36), "군중 밀집도 분석 결과", font=F(30), fill=(255, 255, 255))
rows = [("머리 탐지", f"{len(heads)}개"), ("추정 인원", f"{total:.0f}명"), ("가장 붐빈 2m 칸", f"㎡당 {maxd:.1f}명")]
y = 110
for k, v in rows:
    dr.text((x0, y), k, font=R(22), fill=(170, 185, 205))
    dr.text((x0, y + 30), v, font=F(40), fill=(255, 255, 255))
    y += 110
dr.text((x0, y), "경보 단계", font=R(22), fill=(170, 185, 205))
dr.rounded_rectangle([x0, y + 34, x0 + 160, y + 96], 12, fill=lc)
dr.text((x0 + 80 - dr.textlength(lv, font=F(36)) / 2, y + 38), lv, font=F(36), fill=(255, 255, 255))
y += 130
for name, th, c in [("주의", "㎡당 3명 이상", (240, 200, 0)), ("경계", "㎡당 4명 이상", (255, 140, 0)), ("심각", "㎡당 5명 이상", (220, 40, 40))]:
    dr.rectangle([x0, y + 6, x0 + 20, y + 26], fill=c)
    dr.text((x0 + 32, y), f"{name}  {th}", font=R(22), fill=(225, 230, 240))
    y += 36
dr.text((x0, y + 4), "기준: 서울시 인파 관리 훈련", font=R(18), fill=(150, 165, 185))
note = ["시연용 결과: 머리는 P2PNet, 밀도는", "DM-Count 공개 모델로 대신 구함.", "넓이는 사람 키(1.65m)로 추정, 흰 선 = 분석 구역."]
for i, l in enumerate(note):
    dr.text((x0, H - 100 + i * 26), l, font=R(17), fill=(150, 165, 185))
canvas.save(out, quality=92)
print("max", maxd, lv)
