"""Head locations with P2PNet (ShanghaiTech A weights), drawn as head boxes. CPU.
usage: python3 crowd_heads.py in.jpg out.jpg"""
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
import sys
import numpy as np, cv2, torch, torchvision
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
_vgg = torchvision.models.vgg16_bn
torchvision.models.vgg16_bn = lambda pretrained=False, **k: _vgg(weights=None)  # trained weights overwrite it
torchvision.__version__ = "0.99"  # old version check in util/misc.py
import types; _sp = types.ModuleType("scipy"); _spo = types.ModuleType("scipy.optimize"); _spo.linear_sum_assignment = None; _sp.optimize = _spo; sys.modules.update({"scipy": _sp, "scipy.optimize": _spo})  # matcher only used in training
sys.path.insert(0, REFS + "/CrowdCounting-P2PNet")
import models.vgg_ as _V
_f = _V.vgg16_bn
_V.vgg16_bn = lambda pretrained=False, **k: _f(pretrained=False, **k)  # trained weights overwrite the backbone
from models import build_model

class A: backbone = "vgg16_bn"; row = 2; line = 2
src, out = sys.argv[1], sys.argv[2]
model = build_model(A())
model.load_state_dict(torch.load(REFS + "/CrowdCounting-P2PNet/weights/SHTechA.pth", map_location="cpu")["model"])
model.eval()
im = Image.open(src).convert("RGB")
W, H = im.size
w2, h2 = W // 128 * 128, H // 128 * 128
x = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(im.resize((w2, h2), Image.LANCZOS))
with torch.no_grad():
    o = model(x.unsqueeze(0))
sc = torch.softmax(o["pred_logits"], -1)[0, :, 1]
pts = o["pred_points"][0][sc > 0.5].numpy() * [W / w2, H / h2]
print("heads", len(pts))

img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
sl, yhz = float(sys.argv[3]), float(sys.argv[4])  # person height ~ sl*(y - yhz); head ~ 0.13 of it
for px, py in pts:
    s = max(8, 0.23 * sl * (py - yhz) / (1 - 0.87 * sl))  # head size from the person-height fit
    s = min(s, 60)
    cv2.rectangle(img, (int(px - s / 2), int(py - s / 2)), (int(px + s / 2), int(py + s / 2)), (0, 230, 255), 2, cv2.LINE_AA)
pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
dr = ImageDraw.Draw(pil)
F = lambda n: ImageFont.truetype(FONT_B, n)
dr.rectangle([0, 0, W, 64], fill=(17, 30, 50))
dr.text((20, 8), f"머리 {len(pts)}개 탐지", font=F(36), fill=(255, 220, 0))
note = "머리 위치 예측: P2PNet (ShanghaiTech A 학습 가중치)"
dr.text((W - dr.textlength(note, font=F(22)) - 20, 20), note, font=F(22), fill=(190, 200, 215))
pil.save(out, quality=92)
