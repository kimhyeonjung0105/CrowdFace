"""HT21(CroHD) 머리 정답 → YOLO 라벨 변환 — '현장 영상으로 추가 학습(파인튜닝)'용 데이터.

usage: python src/convert_ht21.py [간격=5]
  학습: HT21-01, HT21-04 전체 + HT21-02 앞 80%   (간격마다 1장)
  검증: HT21-02 뒤 20%                            (best.pt 고르기용)
  시험: HT21-03은 **넣지 않는다** → eval_video.py로 따로 평가 (공정한 시험)
결과: data/ht21_yolo/images|labels/train|val, data/ht21_yolo/ht21.yaml

왜 필요한가
  CrowdHuman(대부분 눈높이 사진)으로만 학습한 head_base가 위에서 내려다본 거리 영상(HT21-03)에서는
  사람을 72%만 찾았다(eval_video.py). 같은 YOLO 학습 코드로 '그 환경' 영상을 조금 더 학습시켜 차이를 줄인다.
gt.txt 클래스 (CroHD 논문): 1 걷는 사람, 2 탈것 위 사람, 3 서 있는 사람, 4 무시 영역
  → 1~3은 모두 '사람 머리'라 라벨로 쓰고, 4는 convert_crowdhuman.py와 같은 이유로 회색(114)으로 칠해 지운다.
간격: 25fps 영상의 이웃 프레임은 거의 같은 사진이라 5프레임마다 1장만 써도 정보는 거의 같고 학습은 5배 빠르다.
"""
import sys, shutil, pathlib as _pl
import numpy as np, cv2
from cvio import imread, imwrite   # 한글 경로에서도 되는 사진 읽기·쓰기 (cv2.imread/imwrite 대신)
ROOT = _pl.Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "ht21" / "HT21" / "train"
OUT = ROOT / "data" / "ht21_yolo"
GRAY = (114, 114, 114)
STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def convert(seq, frames, subset):
    gt = np.loadtxt(SRC / seq / "gt" / "gt.txt", delimiter=",")
    n_head = 0
    for f in frames:
        img = imread(str(SRC / seq / "img1" / f"{f:06d}.jpg")); H, W = img.shape[:2]
        g = gt[gt[:, 0] == f]
        painted = False
        for x, y, w, h in g[g[:, 7] == 4][:, 2:6]:            # 무시 영역 지우기
            cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), GRAY, -1); painted = True
        lines = []
        for x, y, w, h in g[g[:, 7] <= 3][:, 2:6]:
            x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)   # 사진 밖은 잘라냄
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        name = f"{seq}_{f:06d}"
        dst = OUT / "images" / subset / f"{name}.jpg"
        if painted:
            imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            shutil.copy(SRC / seq / "img1" / f"{f:06d}.jpg", dst)
        (OUT / "labels" / subset / f"{name}.txt").write_text("\n".join(lines))
        n_head += len(lines)
    return len(frames), n_head


def main():
    for d in ("images/train", "images/val", "labels/train", "labels/val"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    n02 = len(list((SRC / "HT21-02" / "img1").glob("*.jpg"))); cut = int(n02 * 0.8)
    plan = [("HT21-01", "train", range(1, 430)), ("HT21-04", "train", range(1, 998)),
            ("HT21-02", "train", range(1, cut + 1)), ("HT21-02", "val", range(cut + 1, n02 + 1))]
    for seq, subset, fr in plan:
        n, h = convert(seq, list(fr)[::STEP], subset)
        print(f"{seq} → {subset}: {n}장, 머리 {h:,}개")
    (OUT / "ht21.yaml").write_text(f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: head\n",
                                   encoding="utf-8")
    print("완료:", OUT / "ht21.yaml")


if __name__ == "__main__":
    main()
