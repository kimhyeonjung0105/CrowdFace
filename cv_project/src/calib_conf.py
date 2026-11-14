"""인원 세기용 신뢰도 기준 정하기 — 검증 사진에서 '센 인원 합 = 실제 인원 합'이 되는 값을 찾는다.

usage: python src/calib_conf.py <모델 이름> <데이터 yaml>
  예) python src/calib_conf.py head_base data/head_yolo/head.yaml      → 0.15 (CrowdHuman 검증 4,370장)
      python src/calib_conf.py head_ht21 data/ht21_yolo/ht21.yaml      → HT21-02 뒤 20% (HT21-03 시험 영상은 쓰지 않음)
왜 필요한가: Ultralytics 기본 기준 0.25는 '박스 정확도(mAP)'용이라 인원 세기에는 덜 세는 쪽으로 치우쳤다(87%).
  기준을 시험 데이터가 아닌 검증 데이터에서 정해야 공정하다(시험 영상에 맞춰 고르면 반칙).
"""
import sys, pathlib as _pl
import numpy as np
ROOT = _pl.Path(__file__).resolve().parent.parent
from ultralytics import YOLO


def main():
    name, data = sys.argv[1], ROOT / sys.argv[2]
    # 데이터 yaml은 "키: 값" 몇 줄뿐이라 직접 읽음 (PyYAML이 설치돼 있지 않음)
    d = dict(l.split(": ", 1) for l in data.read_text(encoding="utf-8").splitlines() if ": " in l and not l.startswith(" "))
    base = _pl.Path(d["path"].strip())
    imgs = sorted((base / d["val"].strip()).glob("*.jpg"))
    lbl = lambda p: base / d["val"].strip().replace("images", "labels") / (p.stem + ".txt")
    gt = np.array([len([l for l in lbl(p).read_text().splitlines() if l.strip()]) for p in imgs])
    m = YOLO(str(ROOT / "models" / f"{name}.pt"))
    confs = [m.predict(str(p), imgsz=960, conf=0.03, max_det=1000, verbose=False)[0].boxes.conf.cpu().numpy() for p in imgs]
    best = None
    for c in np.arange(0.05, 0.51, 0.05):
        n = np.array([(x >= c).sum() for x in confs]); ratio = n.sum() / gt.sum()
        print(f"conf {c:.2f}: 찾은 비율 {ratio:6.1%}  MAE {np.abs(n - gt).mean():6.2f}")
        if best is None or abs(ratio - 1) < abs(best[1] - 1):
            best = (round(float(c), 2), ratio)
    print(f"→ {name}: 인원 세기용 기준 {best[0]} (찾은 비율 {best[1]:.1%}, 검증 {len(imgs)}장)")


if __name__ == "__main__":
    main()
