"""DM-Count (QNRF weights) density map overlay. usage: python3 crowd_density.py in.jpg out.jpg"""
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
import numpy as np, cv2, torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
sys.path.insert(0, REFS + "/DM-Count")
import torch.utils.model_zoo as mz; mz.load_url = lambda *a, **k: {}  # ImageNet init not needed
from models import vgg19

src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert("RGB")
W, H = im.size
m = vgg19(); m.load_state_dict(torch.load(REFS + "/DM-Count/pretrained_models/model_qnrf.pth", map_location="cpu")); m.eval()
with torch.no_grad():
    den, _ = m(NORM(transforms.ToTensor()(im)).unsqueeze(0))  # same normalisation as DM-Count training
d = den[0, 0].numpy()
n = d.sum()
print("count", n)
d = cv2.GaussianBlur(cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 3)
d = np.clip(d / np.percentile(d, 99.7), 0, 1)
heat = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
over = cv2.addWeighted(img, 0.55, heat, 0.45, 0)
pil = Image.fromarray(cv2.cvtColor(over, cv2.COLOR_BGR2RGB))
dr = ImageDraw.Draw(pil)
F = lambda s: ImageFont.truetype(FONT_B, s)
dr.rectangle([0, 0, W, 64], fill=(17, 30, 50))
dr.text((20, 8), f"추정 인원 {n:.0f}명", font=F(36), fill=(255, 255, 255))
note = "밀도 지도: DM-Count (UCF-QNRF 학습 가중치)"
dr.text((W - dr.textlength(note, font=F(22)) - 20, 20), note, font=F(22), fill=(190, 200, 215))
# colour bar
bx, by, bw, bh = W - 300, H - 52, 260, 16
bar = cv2.applyColorMap(np.tile(np.linspace(0, 255, bw).astype(np.uint8), (bh, 1)), cv2.COLORMAP_JET)
dr.rounded_rectangle([bx - 14, by - 30, bx + bw + 14, by + bh + 10], 8, fill=(17, 30, 50))
pil.paste(Image.fromarray(cv2.cvtColor(bar, cv2.COLOR_BGR2RGB)), (bx, by))
dr.text((bx, by - 28), "한산", font=F(18), fill=(255, 255, 255))
dr.text((bx + bw - 36, by - 28), "밀집", font=F(18), fill=(255, 255, 255))
pil.save(out, quality=92)
