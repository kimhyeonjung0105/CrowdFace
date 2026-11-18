"""발 위치 가정 확인 (HT21-03): 전신 YOLO(X) 박스의 발과 짝지은 머리로 '머리 아래~발 = 머리 높이의 몇 배'를 잰다.
usage: python src/check_foot.py . <그림.jpg> [가정 배수=6]
결과(2026-11-18): 1,167명, 중앙값 ×4.79(25~75%: 4.33~5.48), 화면 높이와 무관. ×6이면 바닥 오차 중앙값 1.49m, ×4.79면 0.67m.
"""
import sys, json, pathlib, numpy as np, cv2
ROOT = pathlib.Path(sys.argv[1]); OUT = sys.argv[2]; sys.path.insert(0, str(ROOT / "src"))
from cvio import imread, imwrite
from fuse_count import fuse
from crowd_alarm import Floor
from ultralytics import YOLO
seq = ROOT / "data/ht21/HT21/train/HT21-03"; imgs = sorted((seq / "img1").glob("*.jpg"))
gt = np.loadtxt(seq / "gt/gt.txt", delimiter=","); gt = gt[gt[:, 7] <= 3]
cfg = json.loads((seq / "img1.floor.json").read_text(encoding="utf-8")); fl = Floor(cfg["floor"], cfg["size"])
KF = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
body = YOLO(str(ROOT / "models/yolo11n.pt"))
K, Y, D6 = [], [], []
for p in imgs[::25]:
    img = imread(p); H = img.shape[0]
    g = gt[gt[:, 0] == int(p.stem)][:, 2:6]; hb = np.c_[g[:, :2], g[:, :2] + g[:, 2:]]
    bb = body.predict(img, imgsz=1920, conf=0.5, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()
    pairs, _, _ = fuse(hb, bb)
    for h, b in pairs:
        hh = hb[h, 3] - hb[h, 1]
        if bb[b, 3] > H - 5: continue                         # 발이 화면 밖으로 잘린 사람 제외
        k = (bb[b, 3] - hb[h, 3]) / hh                          # 머리 아래 ~ 발 = 머리 높이의 몇 배
        K.append(k); Y.append(hb[h, 3])
        f6 = fl.to_floor([[(hb[h, 0] + hb[h, 2]) / 2, hb[h, 3] + KF * hh]])[0]; fr = fl.to_floor([[(bb[b, 0] + bb[b, 2]) / 2, bb[b, 3]]])[0]
        D6.append(np.hypot(*(f6 - fr)))
K, Y, D6 = map(np.array, (K, Y, D6))
print("짝지은 사람", len(K), " 머리 아래~발 = 머리 높이 × 중앙값 %.2f (25~75%%: %.2f~%.2f)" % (np.median(K), *np.percentile(K, [25, 75])))
for lo, hi in ((0, 400), (400, 700), (700, 1080)):
    s = (Y >= lo) & (Y < hi)
    if s.sum(): print(f"  화면 높이 {lo}~{hi}px: {s.sum()}명, ×{np.median(K[s]):.2f}")
print("×6 가정으로 잡은 발 vs 전신 박스 발: 바닥에서 거리 중앙값 %.2fm (75%%: %.2fm)" % (np.median(D6), np.percentile(D6, 75)))
# 그림: 한 프레임에 ×6 발(주황 점)과 전신 박스 발(파랑 점)
p = imgs[500]; img = imread(p); g = gt[gt[:, 0] == int(p.stem)][:, 2:6]; hb = np.c_[g[:, :2], g[:, :2] + g[:, 2:]]
bb = body.predict(img, imgsz=1920, conf=0.5, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy(); pairs, _, _ = fuse(hb, bb)
for h, b in pairs:
    hh = hb[h, 3] - hb[h, 1]; cx = (hb[h, 0] + hb[h, 2]) / 2
    cv2.rectangle(img, tuple(hb[h, :2].astype(int)), tuple(hb[h, 2:].astype(int)), (52, 104, 235), 2)
    cv2.rectangle(img, tuple(bb[b, :2].astype(int)), tuple(bb[b, 2:].astype(int)), (214, 120, 42), 1)
    cv2.circle(img, (int(cx), int(hb[h, 3] + 6 * hh)), 7, (52, 104, 235), -1)
    cv2.circle(img, (int((bb[b, 0] + bb[b, 2]) / 2), int(bb[b, 3])), 7, (214, 120, 42), -1)
imwrite(OUT, img); print("그림", OUT)
