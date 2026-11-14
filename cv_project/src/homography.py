"""원근 변환(호모그래피) 직접 구현 — 점 4쌍으로 3×3 행렬을 구하고 점을 옮긴다.

usage: python src/homography.py      OpenCV(cv2.getPerspectiveTransform / perspectiveTransform)와 같은 답인지 확인
       다른 코드에서:  from homography import find_homography, apply_homography

원리
  바닥은 평면이라, 사진의 점 (x, y)와 바닥의 점 (X, Y) 사이는 3×3 행렬 H 하나로 정해진다.
      [X·w]   [h1 h2 h3] [x]
      [Y·w] = [h4 h5 h6] [y]        →   X = (h1·x + h2·y + h3) / (h7·x + h8·y + 1)
      [ w ]   [h7 h8  1] [1]             Y = (h4·x + h5·y + h6) / (h7·x + h8·y + 1)
  w로 나누는 것이 '멀수록 작게 보이는' 원근을 만든다(나눗셈이 없으면 평행사변형까지만 표현됨).
  모르는 숫자는 h1~h8 여덟 개(전체 크기는 상관없어 h9 = 1로 고정).
  위 두 식의 분모를 넘기면 h에 대한 1차식이 된다:
      h1·x + h2·y + h3 - h7·x·X - h8·y·X = X
      h4·x + h5·y + h6 - h7·x·Y - h8·y·Y = Y
  점 한 쌍에 식 2개 → 4쌍이면 식 8개 = 모르는 수 8개 → 연립방정식 A·h = b 를 풀면 끝. 학습이 아니라 계산이다.
  (세 점이 한 직선 위에 있으면 식이 겹쳐서 풀리지 않는다 → 네 점은 사각형을 이뤄야 한다.)
"""
import numpy as np


def find_homography(src, dst):
    """src 4점(사진 좌표) → dst 4점(바닥 좌표, m)로 보내는 3×3 행렬"""
    A, b = [], []
    for (x, y), (X, Y) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -x * X, -y * X]); b.append(X)
        A.append([0, 0, 0, x, y, 1, -x * Y, -y * Y]); b.append(Y)
    h = np.linalg.solve(np.array(A, float), np.array(b, float))      # 8원 1차 연립방정식
    return np.append(h, 1.0).reshape(3, 3)


def apply_homography(H, pts):
    """점들 [N,2]에 H 적용: (x, y, 1)에 곱한 뒤 셋째 값 w로 나눈다"""
    pts = np.asarray(pts, float).reshape(-1, 2)
    p = np.c_[pts, np.ones(len(pts))] @ H.T
    return p[:, :2] / p[:, 2:3]


if __name__ == "__main__":
    import cv2
    src = [[462, 200], [1458, 200], [1773.5, 1010], [146.5, 1010]]            # HT21-03 바닥 네 점
    dst = [[0, 0], [10, 0], [10, 28], [0, 28]]                                  # 10m × 28m
    H = find_homography(src, dst); Hcv = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    print("직접 구한 H\n", np.round(H, 6)); print("OpenCV H\n", np.round(Hcv, 6))
    print("행렬 차이 최대:", np.abs(H - Hcv).max())
    pts = np.random.default_rng(0).uniform([0, 200], [1920, 1080], (1000, 2))    # 화면의 아무 점 1,000개
    a = apply_homography(H, pts); c = cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), Hcv).reshape(-1, 2)
    print(f"점 1,000개 변환 차이 최대: {np.abs(a - c).max() * 1000:.3f} mm")
    print("네 점 확인:", np.round(apply_homography(H, src), 6).tolist())
    back = apply_homography(np.linalg.inv(H), a)                                 # 역행렬로 되돌리면 원래 점
    print(f"바닥 → 사진으로 되돌린 차이 최대: {np.abs(back - pts).max():.2e} px")
