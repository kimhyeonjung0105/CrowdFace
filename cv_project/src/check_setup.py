"""다른 PC에서 git clone 한 뒤 빠진 것 점검 — GitHub에 올리지 않은 파일들을 확인하고 받는 방법을 알려 준다.

usage: python src/check_setup.py [백업zip경로]   (.venv 를 만들기 전이면 아무 파이썬으로 실행해도 됨)
  가중치 백업 CrowdFace_models_backup.zip 이 바탕화면·다운로드·프로젝트 폴더에 있으면 경로 없이도 찾아서
  빠진 .pt 파일만 models/ 로 풀어 준다 (이미 있는 파일은 덮어쓰지 않음).
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
# 백업 zip(CrowdFace_models_backup.zip)이 있으면 빠진 .pt를 models/ 로 자동으로 풀어 준다.
# 찾는 곳: 실행할 때 준 경로 → 프로젝트 폴더 → 바탕화면 → 다운로드
BACKUP = "CrowdFace_models_backup.zip"
cands = [_pl.Path(a) for a in sys.argv[1:]] + [ROOT / BACKUP, ROOT.parent / BACKUP,
                                               _pl.Path.home() / "Desktop" / BACKUP,
                                               _pl.Path.home() / "Downloads" / BACKUP]
zp = next((c for c in cands if c.is_file()), None)
if zp:
    import zipfile
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.endswith(".pt") and not (ROOT / "models" / n).exists():
                (ROOT / "models").mkdir(exist_ok=True)
                (ROOT / "models" / n).write_bytes(z.read(n))
                print(f"  [복원] {n}  ← {zp}")
check((ROOT / "models/yolo11n.pt").exists(), "models/yolo11n.pt (COCO 전신, 기존 X)",
      "python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\" 후 생긴 yolo11n.pt 를 models/ 로 옮김")
for k, h in (("head_base", "약 1.3시간"), ("head_p2", "약 2.5~5시간")):
    check((ROOT / f"models/{k}.pt").exists(), f"models/{k}.pt (우리가 학습한 머리 모델)",
          f"백업 {BACKUP}을 바탕화면·다운로드에 두고 다시 실행(자동 복원), 또는 직접 학습: python src\\train_head.py "
          f"{k.split('_')[1]} 30 960 {16 if k == 'head_base' else 8}  ({h}, GPU 필요)")

print("4) 시연 영상 data/ht21/ (HT21 CroHD, CC BY-NC-SA라 GitHub에 없음)")
h3 = ROOT / "data/ht21/HT21/train/HT21-03"
check(count(h3 / "img1") == 1000, f"HT21-03 프레임 (지금 {count(h3 / 'img1'):,}장 / 1,000장)", "CLAUDE.md 8절의 HT21 다운로드", need="시연용")
check((h3 / "img1.floor.json").exists(), "HT21-03 바닥 설정 img1.floor.json", "CLAUDE.md 8절의 floor 값으로 만들기", need="시연용")
check((ROOT / "models/head_ht21.pt").exists(), "models/head_ht21.pt (현장 영상 추가 학습)",
      "백업 zip 또는 python src/convert_ht21.py → python src/train_head.py ht21 20 960 8", need="시연용")

print("5) 데이터 data/ (학습·평가할 때만 필요, CrowdHuman 재배포 금지라 GitHub에 없음)")
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
