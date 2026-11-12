"""다른 PC에서 git clone 한 뒤 빠진 것 점검 — GitHub에 올리지 않은 파일들을 확인하고 받는 방법을 알려 준다.

usage: python src/check_setup.py        (.venv 를 만들기 전이면 아무 파이썬으로 실행해도 됨)
왜 빠져 있나: 용량(가중치·데이터)·라이선스(CrowdHuman 재배포 금지) 때문에 .gitignore로 뺐다.
자세한 받는 방법은 CLAUDE.md 8절 참고.
"""
import sys, importlib.util, pathlib as _pl
ROOT = _pl.Path(__file__).resolve().parent.parent
miss = 0


def check(ok, name, fix, need="필수"):
    global miss
    print(f"  [{'OK' if ok else '없음'}] {name}" + ("" if ok else f"   ({need})\n         → {fix}"))
    if not ok and need == "필수":
        miss += 1


def count(p, pat="*.jpg"):
    return sum(1 for _ in p.glob(pat)) if p.exists() else 0


print("1) 파이썬 환경")
check(sys.version_info[:2] == (3, 13), f"Python 3.13 (지금 {sys.version.split()[0]})",
      "py -3.13 -m venv .venv  →  .venv\\Scripts\\activate", need="권장")
pk = {"torch": "torch", "ultralytics": "ultralytics", "cv2": "opencv-python", "reportlab": "reportlab",
      "matplotlib": "matplotlib"}
for mod, pip in pk.items():
    check(importlib.util.find_spec(mod) is not None, f"패키지 {pip}", "pip install -r requirements.txt")
if importlib.util.find_spec("torch"):
    import torch
    check(torch.cuda.is_available(), f"GPU(CUDA) torch (지금 {torch.__version__})",
          "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132  (학습할 때만 필요)",
          need="학습용")

print("2) 참조 저장소 refs/ (기존 시연 코드 crowd_*.py 용)")
check((ROOT / "refs/DM-Count/pretrained_models").exists(), "refs/DM-Count",
      "git clone https://github.com/cvlab-stonybrook/DM-Count refs/DM-Count")
check((ROOT / "refs/CrowdCounting-P2PNet/weights").exists(), "refs/CrowdCounting-P2PNet",
      "git clone https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet refs/CrowdCounting-P2PNet")

print("3) 가중치 models/")
check((ROOT / "models/yolo11n.pt").exists(), "models/yolo11n.pt (COCO 전신, 기존 X)",
      "python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\" 후 생긴 yolo11n.pt 를 models/ 로 옮김")
for k, h in (("head_base", "약 1.3시간"), ("head_p2", "약 2.5~5시간")):
    check((ROOT / f"models/{k}.pt").exists(), f"models/{k}.pt (우리가 학습한 머리 모델)",
          f"팀 드라이브에서 받아 models/ 에 넣기, 또는 직접 학습: python src\\train_head.py "
          f"{k.split('_')[1]} 30 960 {16 if k == 'head_base' else 8}  ({h}, GPU 필요)")

print("4) 데이터 data/ (학습·평가할 때만 필요, CrowdHuman 재배포 금지라 GitHub에 없음)")
ch = ROOT / "data/crowdhuman"
check((ch / "annotation_val.odgt").exists() and (ch / "annotation_train.odgt").exists(),
      "data/crowdhuman/annotation_train.odgt · annotation_val.odgt", "CLAUDE.md 8절의 다운로드 명령", need="학습용")
n = count(ch / "Images")
check(n >= 19370, f"data/crowdhuman/Images (지금 {n:,}장 / 19,370장)", "CLAUDE.md 8절 (train01~03 + val zip 풀기)",
      need="학습용")
for s, want in (("train", 15000), ("val", 4370)):
    n = count(ROOT / f"data/head_yolo/images/{s}")
    check(n == want, f"data/head_yolo/{s} (지금 {n:,}장 / {want:,}장)",
          f"python src\\convert_crowdhuman.py data\\crowdhuman\\annotation_{s}.odgt data\\crowdhuman\\Images {s}",
          need="학습용")

print("\n" + ("필수 항목은 모두 있음. 시연·평가 코드를 돌릴 수 있다." if miss == 0
              else f"필수 항목 {miss}개가 없음. 위 → 안내대로 받은 뒤 다시 실행."))
