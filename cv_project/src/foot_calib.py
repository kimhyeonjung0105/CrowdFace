"""발 위치 자동 보정: 전신 YOLO(기존 기능 X)로 '머리 아래 ~ 발 = 머리 높이의 몇 배'를 카메라마다 잰다.

왜 필요한가 (check_foot.py, HT21-03)
  - 원근 변환은 바닥 위 점에만 맞으므로 머리(바닥에서 1.6m 위) 대신 발 위치를 바닥으로 옮겨야 한다.
  - 처음에는 '사람 키 ≈ 머리 7개'라는 가정으로 머리 높이 × 6 아래를 발로 잡았다(확인 안 한 가정).
  - 발까지 보이는 사람의 전신 박스로 재 보니 정답 머리 박스 기준 × 4.79, 우리 모델 박스 기준 × 5.47.
    ×6이면 바닥에서 1.5m(2m 칸 거의 한 칸) 어긋났다.
  - 배수는 카메라 각도와 '머리 박스를 얼마나 크게 그리느냐'에 따라 달라진다 → 고정값이 아니라 그 카메라·그 탐지기로 잰다.
방법
  1) 영상에서 프레임 몇 장을 고르게 뽑아 머리 탐지와 전신 탐지(YOLO 전신, 입력 1920)를 함께 돌린다
  2) 머리 중심이 전신 박스 위쪽 1/3 안이면 같은 사람으로 짝짓는다(fuse_count.fuse, 가까운 짝부터 1:1)
  3) 발이 화면 아래로 잘린 사람, 머리가 너무 작은 사람(4px 미만), 비율이 말이 안 되는 짝(1.5배 미만·12배 초과)은 뺀다
  4) 남은 사람들의 (전신 박스 아래 - 머리 박스 아래) ÷ 머리 높이의 중앙값 = 발 배수
     (다리가 앞사람에 가려 박스가 짧아지는지 확인: 겹치지 않는 사람만 써도 4.74로 같았다)
"""
import sys, pathlib as _pl
import numpy as np
ROOT = _pl.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from fuse_count import fuse

MIN_PAIRS = 30          # 짝이 이보다 적으면 믿지 않고 기본값을 쓴다
DEFAULT = 6.0           # 보정 실패 때 쓰는 예전 가정값


def foot_ratios(head_boxes, body_boxes, img_h):
    """한 프레임: 짝지은 사람마다 (머리 아래 ~ 발) ÷ 머리 높이"""
    out = []
    for h, b in fuse(head_boxes, body_boxes)[0]:
        hh = head_boxes[h, 3] - head_boxes[h, 1]
        if body_boxes[b, 3] > img_h - 5 or hh < 4:
            continue
        r = (body_boxes[b, 3] - head_boxes[h, 3]) / hh
        if 1.5 < r < 12:
            out.append(r)
    return out


def calibrate_foot(frames, head_fn, body_model=None, body_imgsz=1920, body_conf=0.5):
    """frames: 사진(BGR) 목록, head_fn(img) → 머리 박스 [N,4] → (배수, 짝 수). 짝이 모자라면 (DEFAULT, 짝 수)"""
    if body_model is None:
        from ultralytics import YOLO
        body_model = YOLO(str(ROOT / "models" / "yolo11n.pt"))
    rs = []
    for img in frames:
        bb = body_model.predict(img, imgsz=body_imgsz, conf=body_conf, classes=[0], verbose=False)[0].boxes.xyxy.cpu().numpy()
        rs += foot_ratios(np.asarray(head_fn(img)), bb, img.shape[0])
    if len(rs) < MIN_PAIRS:
        return DEFAULT, len(rs)
    return float(np.median(rs)), len(rs)
