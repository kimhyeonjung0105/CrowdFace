"""기존 기능 X: COCO 사전학습 YOLO로 사람 '전신' 박스를 찾아 인원 세기 (코드 수정 없음).
머리 탐지(추가 기능 Y)와 비교하기 위한 기준선.
usage: python src/yolo_person.py in.jpg out.jpg [imgsz=640]
VS Code ▶ 버튼으로 인자 없이 실행하면 아래 기본값을 씀."""
import sys, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parent.parent
MODELS = str(ROOT / "models")
import cv2
from ultralytics import YOLO

# ▶ 실행용 기본값 — 사진·입력 크기를 바꾸려면 여기만 고치면 됨
SRC = str(ROOT / "refs/DM-Count/example_images/1.png")
OUT = str(ROOT / "results/yolo_person.jpg")
IMGSZ = 640  # Ultralytics 기본값. 밀집 장면은 1920 추천

src = sys.argv[1] if len(sys.argv) > 1 else SRC
out = sys.argv[2] if len(sys.argv) > 2 else OUT
imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else IMGSZ

r = YOLO(MODELS + "/yolo11n.pt").predict(src, classes=[0], conf=0.25, imgsz=imgsz, verbose=False)[0]
n = len(r.boxes)
img = r.plot(labels=False, conf=False, line_width=2)
cv2.rectangle(img, (0, 0), (520, 60), (0, 0, 0), -1)
cv2.putText(img, f"YOLO person: {n}  (imgsz {imgsz})", (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
cv2.imwrite(out, img)
print("persons", n, "imgsz", imgsz)
print("결과 사진:", out)
import os
if hasattr(os, "startfile"): os.startfile(out)  # Windows: 사진 뷰어로 바로 열기
