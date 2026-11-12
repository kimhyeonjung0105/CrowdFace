"""머리 탐지 학습 (추가 기능 Y①) — 이 PC의 GPU(RTX 4070)로 학습.

usage: python src/train_head.py base|p2 [epochs=30] [imgsz=960] [batch=16]
  base : 원본 YOLO11n 구조. yolo11n.pt(COCO)를 불러오면 Ultralytics가 Detect 층의 클래스 출력만
         80 → 1('head')로 새로 만들고, 나머지 층은 COCO 가중치에서 시작한다(전이학습).
  p2   : 우리가 직접 수정한 구조(models/head_p2.yaml, 작은 머리용 P2 탐지층 추가).
         구조를 만든 뒤 yolo11n.pt에서 모양이 맞는 층(백본 등)만 옮겨 받는다(297/593 항목).
결과: runs/head/head_<base|p2>/weights/best.pt  → 끝나면 models/head_<base|p2>.pt 로 복사됨
중간에 끊기면 같은 명령에 resume을 붙여 이어서 학습: python src/train_head.py base resume

두 모델을 공정하게 비교하려면 epochs·imgsz·batch를 똑같이 쓴다.
"""
import sys, shutil, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data" / "head_yolo" / "head.yaml"
PROJECT = ROOT / "runs" / "head"
from ultralytics import YOLO


def main():
    kind = sys.argv[1]                                   # base | p2
    name = f"head_{kind}"
    last = PROJECT / name / "weights" / "last.pt"

    if len(sys.argv) > 2 and sys.argv[2] == "resume":    # 끊긴 학습 이어서
        YOLO(str(last)).train(resume=True)
    else:
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else 960   # 머리가 작아서 기본 640보다 크게
        batch = int(sys.argv[4]) if len(sys.argv) > 4 else 16    # GPU 메모리 부족하면 8
        if kind == "base":
            model = YOLO(str(MODELS / "yolo11n.pt"))
        else:
            model = YOLO(str(MODELS / "head_p2.yaml")).load(str(MODELS / "yolo11n.pt"))
        model.train(data=str(DATA), epochs=epochs, imgsz=imgsz, batch=batch, device=0,
                    workers=4, project=str(PROJECT), name=name, exist_ok=True, seed=0)

    best = PROJECT / name / "weights" / "best.pt"
    shutil.copy(best, MODELS / f"{name}.pt")
    print("완료:", MODELS / f"{name}.pt")


if __name__ == "__main__":   # Windows에서 데이터 로더(멀티프로세스)를 쓰려면 꼭 필요
    main()
