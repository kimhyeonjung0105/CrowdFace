"""전신(X) + 머리(Y) 결합 세기 — 같은 사람의 전신 박스와 머리 박스를 짝지어 한 명으로 센다.

usage: python src/fuse_count.py in.jpg out.jpg [머리모델=head_base] [머리conf=0.15] [전신conf=0.5]
       다른 코드에서:  from fuse_count import fuse

왜 필요한가
  - 전신 YOLO는 한산할 때 잘 찾고 붐비면 놓친다. 머리 모델은 붐벼도 찾지만 뒤돌아선 검은 머리 등을 놓친다.
  - 둘을 그냥 더하면 같은 사람을 두 번 센다 → '이 머리가 이 몸의 머리인가'를 판단해 짝지어야 한다.
규칙 (직접 구현)
  1) 짝 후보: 머리 박스 중심이 전신 박스의 가로 범위 안 + 세로로 위쪽 1/3 안 (서 있는 사람의 머리는 몸 맨 위에 있으므로)
  2) 붐비면 한 머리가 여러 전신 박스의 위쪽 1/3에 들어간다(앞사람 머리가 뒷사람 몸 박스 안).
     → 후보마다 '머리 중심 ↔ 전신 박스 윗변 가운데' 거리(전신 폭으로 나눔)를 구해 **가까운 짝부터** 1:1로 확정한다.
  3) 인원 = 짝(머리+전신) + 머리만 + 전신만
     머리만: 몸이 가려진 사람(붐빌 때 대부분)   전신만: 머리를 놓친 사람(뒤통수·모자·멀리)
"""
import sys, pathlib as _pl
import numpy as np
ROOT = _pl.Path(__file__).resolve().parent.parent
TOP = 1 / 3        # 전신 박스의 위쪽 이 비율 안에 머리 중심이 있어야 같은 사람


def fuse(heads, bodies):
    """heads [N,4], bodies [M,4] (xyxy) → (pairs [(머리 번호, 전신 번호)], 머리만 번호들, 전신만 번호들)"""
    n, m = len(heads), len(bodies)
    if n == 0 or m == 0:
        return [], np.arange(n), np.arange(m)
    cx = (heads[:, 0] + heads[:, 2]) / 2; cy = (heads[:, 1] + heads[:, 3]) / 2
    bw = bodies[:, 2] - bodies[:, 0]; bh = bodies[:, 3] - bodies[:, 1]
    inside = (cx[:, None] >= bodies[None, :, 0]) & (cx[:, None] <= bodies[None, :, 2]) & \
             (cy[:, None] >= bodies[None, :, 1]) & (cy[:, None] <= bodies[None, :, 1] + TOP * bh[None, :])     # 1) 후보 [N,M]
    hi, bi = np.nonzero(inside)
    bx = (bodies[:, 0] + bodies[:, 2]) / 2
    dist = np.hypot(cx[hi] - bx[bi], cy[hi] - bodies[bi, 1]) / np.maximum(bw[bi], 1)                           # 2) 윗변 가운데까지 거리
    used_h, used_b, pairs = np.zeros(n, bool), np.zeros(m, bool), []
    for k in np.argsort(dist):                                      # 가까운 짝부터 확정, 이미 짝이 있는 머리·전신은 건너뜀
        h, b = hi[k], bi[k]
        if not used_h[h] and not used_b[b]:
            used_h[h] = used_b[b] = True; pairs.append((int(h), int(b)))
    return pairs, np.nonzero(~used_h)[0], np.nonzero(~used_b)[0]


def count(heads, bodies):
    p, ho, bo = fuse(heads, bodies)
    return len(p) + len(ho) + len(bo)


if __name__ == "__main__":
    import cv2
    from cvio import imread, imwrite
    from ultralytics import YOLO
    src, dst = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "head_base"
    ch = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
    cb = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
    img = imread(src)
    H = YOLO(str(ROOT / "models" / f"{name}.pt")).predict(img, imgsz=960, conf=ch, max_det=1000, verbose=False)[0].boxes.xyxy.cpu().numpy()
    B = YOLO(str(ROOT / "models" / "yolo11n.pt")).predict(img, imgsz=960, conf=cb, classes=[0], max_det=1000, verbose=False)[0].boxes.xyxy.cpu().numpy()
    pairs, ho, bo = fuse(H, B)
    box = lambda b, c, t=2: cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, t)
    for h, b in pairs:                       # 짝: 초록 (전신은 가늘게)
        box(B[b], (122, 175, 27), 1); box(H[h], (122, 175, 27))
    for h in ho: box(H[h], (52, 104, 235))   # 머리만: 주황
    for b in bo: box(B[b], (214, 120, 42))   # 전신만: 파랑
    imwrite(dst, img)
    print(f"머리 {len(H)} · 전신 {len(B)} → 짝 {len(pairs)} + 머리만 {len(ho)} + 전신만 {len(bo)} = {len(pairs) + len(ho) + len(bo)}명  ({dst})")
