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
import sys, torch, numpy as np, cv2
from PIL import Image
from torchvision import transforms
sys.path.insert(0, REFS + "/DM-Count")
import torch.utils.model_zoo as mz; mz.load_url = lambda *a, **k: {}  # ImageNet 초기값 불필요(학습된 가중치로 덮어씀)
from models import vgg19
from ultralytics import YOLO

imgs = sys.argv[1:] or [REFS + "/DM-Count/example_images/1.png",REFS + "/DM-Count/example_images/2.png",
        REFS + "/DM-Count/example_images/3.png",REFS + "/CrowdCounting-P2PNet/vis/demo1.jpg"]
yolo = YOLO(MODELS + "/yolo11n.pt")
dm = vgg19(); dm.load_state_dict(torch.load(REFS + "/DM-Count/pretrained_models/model_qnrf.pth", map_location="cpu")); dm.eval()
for i, p in enumerate(imgs):
    im = Image.open(p).convert("RGB")
    r = yolo.predict(p, classes=[0], conf=0.25, verbose=False)[0]
    ny = len(r.boxes)
    with torch.no_grad():
        den, _ = dm(transforms.ToTensor()(im).unsqueeze(0))
    nd = den.sum().item()
    print(f"{p.split('/')[-1]:12s} size={im.size} YOLO={ny:4d}  DM-Count={nd:7.1f}")
    # save side-by-side visual
    a = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    yb = r.plot(labels=False, conf=False, line_width=1)
    d = den[0,0].numpy(); d = (d-d.min())/(d.max()-d.min()+1e-6)
    d = cv2.applyColorMap((d*255).astype(np.uint8), cv2.COLORMAP_JET); d = cv2.resize(d, (a.shape[1], a.shape[0]))
    over = cv2.addWeighted(a, 0.5, d, 0.5, 0)
    cv2.putText(yb, f"YOLO: {ny}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    cv2.putText(over, f"DM-Count: {nd:.0f}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
    cv2.imwrite(f"{RESULTS}/cmp_{i}.jpg", np.hstack([yb, over]))
