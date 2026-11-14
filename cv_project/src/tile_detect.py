"""타일링 탐지: 사진을 조각내 확대해서 머리를 찾고, 겹친 박스를 합친다 — 작은 머리 대책.

usage: python src/tile_detect.py in.jpg out.jpg [격자=2] [모델=head_base] [conf=0.15]
       다른 코드에서:  from tile_detect import detect_tiled

왜 필요한가
  - 모델은 사진을 imgsz 960으로 줄여서 본다. 1920×1080 영상이면 절반으로 줄어 16px 머리가 8px이 된다.
    P2 탐지층을 넣어도 16px 미만 머리 재현율이 그대로였던 이유 = 입력에서 이미 정보가 사라짐(보고서 4-2).
  - 그래서 모델·가중치는 그대로 두고 **입력을 바꾼다**: 사진을 n×n 조각으로 잘라 조각마다 960으로 넣으면
    같은 머리가 n배 가까이 크게 들어간다.
방법 (직접 구현)
  1) 조각 나누기: n×n, 이웃 조각과 OVERLAP(20%)만큼 겹치게 → 경계에 걸린 머리도 어느 한 조각에는 온전히 들어감
  2) 조각마다 탐지 → 박스 좌표에 조각의 왼쪽 위 좌표를 더해 원본 좌표로 되돌림
  3) 조각 안쪽 경계에 닿은 박스는 버림: 머리가 잘린 채 잡힌 것(반쪽 박스)이라 겹침 영역의 이웃 조각 박스를 믿는다
  4) 전체 사진 탐지도 한 번 더함: 조각보다 큰 머리(가까운 사람)는 어느 조각에서도 잘리므로
  5) 겹친 박스 합치기(NMS): 같은 머리가 여러 조각 + 전체 탐지에서 중복으로 잡힌 것을 신뢰도 높은 것 하나만 남김
"""
import sys, pathlib as _pl
import numpy as np
ROOT = _pl.Path(__file__).resolve().parent.parent
OVERLAP = 0.2      # 이웃 조각과 겹치는 비율
EDGE = 2           # 조각 경계에서 이 픽셀 안에 닿은 박스는 '잘린 박스'로 봄
NMS_IOU = 0.5      # 두 박스가 이만큼 겹치면 같은 머리


def tiles(W, H, n, overlap=OVERLAP):
    """n×n 조각의 (x0, y0, x1, y1) 목록. 조각 크기 t는 n·t - (n-1)·겹침 = 전체 길이가 되게 정한다."""
    tw, th = int(np.ceil(W / (n - (n - 1) * overlap))), int(np.ceil(H / (n - (n - 1) * overlap)))
    xs = np.linspace(0, W - tw, n).round().astype(int); ys = np.linspace(0, H - th, n).round().astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def iou_one(b, B):
    """박스 하나 b와 여러 박스 B의 IoU (겹친 넓이 ÷ 합친 넓이)"""
    x1 = np.maximum(b[0], B[:, 0]); y1 = np.maximum(b[1], B[:, 1])
    x2 = np.minimum(b[2], B[:, 2]); y2 = np.minimum(b[3], B[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter + 1e-9)


def nms(boxes, conf, thr=NMS_IOU):
    """겹친 박스 합치기: 신뢰도가 가장 높은 박스를 남기고, 그것과 thr 이상 겹치는 박스를 지우기를 반복. 남길 번호를 돌려줌."""
    order = np.argsort(-conf); keep = []
    while len(order):
        i = order[0]; keep.append(i)
        rest = order[1:]
        order = rest[iou_one(boxes[i], boxes[rest]) < thr]
    return np.array(keep, int)


def detect_tiled(model, img, n=2, imgsz=960, conf=0.15, full=True, **kw):
    """img(BGR 배열) → (박스 [N,4] xyxy 원본 좌표, 신뢰도 [N]).  n=1이면 보통 탐지와 같다."""
    H, W = img.shape[:2]
    boxes, confs = [], []
    if full or n == 1:                                               # 4) 전체 사진 탐지
        r = model.predict(img, imgsz=imgsz, conf=conf, max_det=1000, verbose=False, **kw)[0]
        boxes.append(r.boxes.xyxy.cpu().numpy()); confs.append(r.boxes.conf.cpu().numpy())
    if n > 1:
        ts = tiles(W, H, n)
        rs = model.predict([img[y0:y1, x0:x1] for x0, y0, x1, y1 in ts], imgsz=imgsz, conf=conf, max_det=1000,
                           verbose=False, **kw)                      # 조각들을 한 묶음(batch)으로 탐지
        for (x0, y0, x1, y1), r in zip(ts, rs):
            b = r.boxes.xyxy.cpu().numpy(); c = r.boxes.conf.cpu().numpy()
            tw, th = x1 - x0, y1 - y0
            # 3) 조각의 '안쪽' 경계(사진 가장자리가 아닌 쪽)에 닿은 박스는 잘린 머리 → 버림
            cut = ((b[:, 0] <= EDGE) & (x0 > 0)) | ((b[:, 1] <= EDGE) & (y0 > 0)) | \
                  ((b[:, 2] >= tw - EDGE) & (x1 < W)) | ((b[:, 3] >= th - EDGE) & (y1 < H))
            boxes.append(b[~cut] + [x0, y0, x0, y0]); confs.append(c[~cut])   # 2) 원본 좌표로
    boxes = np.concatenate(boxes); confs = np.concatenate(confs)
    if n > 1 and len(boxes):
        k = nms(boxes, confs); boxes, confs = boxes[k], confs[k]     # 5) 중복 합치기
    return boxes, confs


if __name__ == "__main__":
    import cv2
    from cvio import imread, imwrite
    from ultralytics import YOLO
    src, dst = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    name = sys.argv[4] if len(sys.argv) > 4 else "head_base"
    conf = float(sys.argv[5]) if len(sys.argv) > 5 else 0.15
    m = YOLO(str(ROOT / "models" / f"{name}.pt")); img = imread(src)
    b1, _ = detect_tiled(m, img, 1, conf=conf); b, _ = detect_tiled(m, img, n, conf=conf)
    for x0, y0, x1, y1 in tiles(img.shape[1], img.shape[0], n):
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (200, 200, 200), 1)       # 조각 경계
    for x0, y0, x1, y1 in b.astype(int):
        cv2.rectangle(img, (x0, y0), (x1, y1), (52, 104, 235), 2)
    imwrite(dst, img)
    print(f"보통 탐지 {len(b1)}명 → 타일링 {n}×{n} {len(b)}명  ({dst})")
