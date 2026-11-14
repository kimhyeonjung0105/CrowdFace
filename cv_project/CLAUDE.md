# CrowdFace: 군중 밀집도 기반 인파 사고 조기 경보 — 프로젝트 안내

> 이 파일은 VS Code에서 작업할 때(사람·AI 도우미 모두) 먼저 읽는 안내서입니다.
> 답변과 주석은 **한국어**로, 짧은 목록 위주로 씁니다.

## 1. 과제 개요
- 프로젝트 이름 **CrowdFace** (GitHub 저장소 이름도 같게). 학습 데이터셋 CrowdHuman과 헷갈리지 말 것.
- 강남대학교 컴퓨터비전 미니프로젝트 (202104183 김현중, 202204201 김태섭)
- 일정: 5주차(10/16) 계획서(주제) 발표 완료 — 중간고사 전으로 당겨서 진행, **별도 중간 발표 없음** → **14주차 최종 발표·시연(12월 셋째 주쯤)**
- 과제 조건: 기존 구현 **X**에 반드시 **추가 기능 Y**를 붙여야 함. 참조 코드는 본인 GitHub에 올려 주소 제출, 발표 PDF는 이러닝 제출.
- **과제 목적: 기존 코드를 다르게 적용하며 컴퓨터비전을 공부하는 것.** 새 기술을 가져다 붙이는 게 아니라 기존 기능을 다른 방식으로 쓰는 것이 중요함 (공지 예: 마스크 판별 코드 → 썬글라스 추가).
  - 핵심 Y = **전신 탐지 → 머리 탐지** (같은 YOLO 학습 코드로 직접 학습·평가. 공개 머리 가중치를 그냥 쓰지 않음).
  - 응용 = ㎡당 밀도·경보 (머리를 세야 하는 이유를 보여 줌. 발표 때 교수님이 이걸 만들지 질문하셨음).
  - 발표 때 교수님 지적: "머리 코드는 있을 것 같은데?" → 직접 학습 과정과 **전신 대비 머리의 인원 오차 비교**(붐빌수록 차이)로 의미를 보여 줄 것.
- **강조점: 우리가 코드를 이해하고 직접 바꿔 학습시켜 머리를 세게 했다.** 보여 줄 것:
  1. CrowdHuman `.odgt` → YOLO 라벨 변환 코드를 직접 작성 (어떤 박스를 왜 뺐는지 포함)
  2. 모델 구조 이해: backbone(COCO 가중치 유지, 전이학습) vs Detect 층(80 클래스 → 'head' 1 클래스로 교체)
  3. (추천) 작은 머리용 P2 탐지층 추가 — 모델 yaml 수정, 추가 전후 비교
  4. 결과: 머리 mAP, 전신 대비 인원 오차
- 작업 원칙: 코드마다 **왜 그렇게 했는지** 한국어 주석·설명을 남겨, 팀원이 발표 질문에 직접 답할 수 있게 한다.

## 2. 동기
- 2022-10-29 이태원 골목은 ㎡당 7.7~10.7명까지 밀집해 159명이 숨짐 (국과수 분석, 서울신문 2023-01-13). 첫 112 신고는 참사 약 3시간 41분 전.
- 서울시는 ㎡당 **3명 주의 · 4명 경계 · 5명 심각**을 기준으로 쓰고(서울시 미디어허브 2023-10-25), 인파감지 CCTV 909대를 설치 계획. 하지만 사람이 화면을 보고 세기는 어렵고, 붐빌수록 몸이 가려져 셀 수 없음.
- 목표: CCTV 영상에서 **머리**를 찾아 인원을 세고, 바닥 넓이로 나눠 ㎡당 인원을 구해 단계별 경보.

## 3. 시스템 구조 (X + Y)
| 단계 | 내용 | 구분 |
|---|---|---|
| 입력 | 높은 곳에 고정한 CCTV·휴대폰 영상 | |
| X 사람 탐지 | COCO 사전학습 YOLO(Ultralytics)로 전신 박스 (코드 수정 없음) | 기존 기능 |
| Y① 머리 탐지 | 같은 YOLO 학습 코드로 CrowdHuman 머리 박스 학습 → '머리' 클래스 추가 (모자·후드·뒤통수 포함) | 추가 기능 ① |
| Y② 밀도·경보 | 바닥 네 점 원근 변환 → 2m×2m 칸별 ㎡당 인원 → 3/4/5명 단계, 같은 단계 5초 지속 시 경보 | 추가 기능 ② |
| 출력 | 칸별 색 표시 영상, 경보 단계·시각·인원 기록 | |

- 전신·머리 연결 규칙: 머리 박스 중심이 전신 박스 위쪽 1/3 안이면 같은 사람. 머리만 잡힌 사람도 1명으로 셈.
- DM-Count(밀도 지도)는 초밀집 장면 **비교용**, P2PNet은 머리 모델 학습 전 **임시 대체**.

## 4. 진행 계획
| 주차 | 할 일 | 완료 기준 |
|---|---|---|
| 9주 | 이 PC에서 결과 사진 3장(밀도 지도·머리·최종) + YOLO 전신 탐지 재현 | `results/`에 사진 |
| 10~11주 | CrowdHuman 머리 라벨(.odgt) → YOLO 라벨 변환, `model.train`으로 머리 학습 | 검증 mAP 수치, `models/head_best.pt` |
| 12주 | 원근 변환(바닥 4점 클릭) + 2m 칸 밀도 + 5초 경보 로직, 영상 입력 | 영상 결과 파일 |
| 13주 | 평가: ShanghaiTech MAE, 위험 단계 정답 비교, CPU 초당 프레임 | 평가 표 |
| 14주 (최종) | 영상 시연, GitHub 정리(README·코드) | 시연 영상, 저장소 주소 |

## 5. 폴더와 코드
```
src/yolo_person.py    기존 기능 X: YOLO 전신 탐지·인원 (in.jpg out.jpg [imgsz])
src/crowd_density.py  DM-Count 밀도 지도          (in.jpg out.jpg)
src/crowd_result.py   YOLO 전신 + 밀도, slope·horizon 출력 (in.jpg out.jpg)
src/crowd_heads.py    P2PNet 머리 위치              (in.jpg out.jpg slope horizon)
src/crowd_final.py    머리+밀도+2m 칸+경보 패널     (in.jpg out.jpg slope horizon)
src/crowd_alarm.py    ★ 최종 시연: 우리 머리 모델 + 바닥 4점 원근 변환 + 2m 칸 + 5초 경보 (사진·영상, --floor/--size 또는 클릭)
src/compare.py        YOLO vs DM-Count 비교 → results/cmp_*.jpg
src/convert_crowdhuman.py  CrowdHuman .odgt → YOLO 머리 라벨 (data/head_yolo/)
models/head_p2.yaml   직접 수정한 구조: YOLO11n + P2 탐지층, nc=1
notebooks/train_head_colab.ipynb  Colab 학습 (예비용, 실제 학습은 이 PC GPU)
src/train_head.py     이 PC GPU 학습 (base|p2 [epochs imgsz batch] / resume) → models/head_<base|p2>.pt
src/eval_heads.py     전신 vs 머리 인원 오차·머리 크기별 재현율 (CrowdHuman val) → results/eval/
src/bench_speed.py    세 모델 번갈아 속도 재측정 → summary.json speed_ms 덮어씀 (eval 뒤에 실행)
src/make_report.py    학습 결과 PDF → results/head_detection_report.pdf
src/check_setup.py    다른 PC에서 clone 후 빠진 것 점검 (환경·refs·가중치·데이터) → 8절
data/   입력  models/ 가중치  refs/ 참조 저장소(git clone)  results/ 결과
```
- 경로는 모두 cv_project 폴더 기준(ROOT). 새 코드도 이 방식을 따른다.
- 새 기능은 `src/`에 파일을 추가하고, 기존 시연용 스크립트는 깨지 않게 둔다.

## 6. 주의사항
**데이터·라이선스**
- CrowdHuman: 비상업 연구·교육용(CC BY-NC 4.0), **이미지 재배포 금지** → GitHub에 이미지·라벨 올리지 말 것. 구글 드라이브 원본은 삭제됨 → Hugging Face `sshao0516/CrowdHuman` 사용. 전체 14.2GB이니 **val.zip + annotation_val.odgt부터** 받아 시험.
- P2PNet 가중치는 학술 전용. 최종 결과는 직접 학습한 머리 YOLO로 바꾸는 것이 목표.
- Ultralytics는 AGPL-3.0. Owen718/Head-Detection-Yolov8은 라이선스 표기가 없어 **참고만**.

**결과 해석**
- 지금 결과 사진은 **개념 시연**: 바닥 넓이를 사람 키 1.65m·화각 60° 가정으로 추정함. 실제 시연은 바닥 4점 원근 변환으로 바꿀 것.
- 전처리에 민감함: DM-Count 정규화 여부만으로 최대 칸이 ㎡당 1.7(정상) ↔ 3.2(주의)로 바뀌었음. 학습 때와 같은 정규화를 꼭 쓸 것.
- YOLO 입력 크기: 군중 사진에서 imgsz 640이면 7명, 1920이면 73명. 밀집 장면은 큰 입력 필요.
- 화면 가장자리에 걸친 칸은 넓이가 작아 가짜 고밀도가 나옴 → 분석 구역(ROI) 안 칸만 계산.

**환경**
- CPU만 쓰면 학습은 사실상 불가 → 머리 학습은 GPU PC(RTX 4070)에서 했다. Colab 노트북은 예비용.
- `.venv`, `refs/`, `models/*.pt`, `data/`의 큰 파일은 Git에 올리지 않는다(.gitignore).
- 발표 수치는 출처가 확인된 것만 쓴다(위 동기 출처 참고).

## 7. 현재 상태 (2026-11-14)
- 코드 5개와 결과 사진(밀도 지도 461명, 머리 345개, 최종 ㎡당 1.7명 정상)은 개발 서버에서 실행 확인.
- 이 PC(Windows)에서도 5개 모두 실행 확인 (2026-11-11): `.venv`는 Python 3.13(3.14는 오래된 참조 코드 호환 우려로 피함), torch 2.15 CPU, Ultralytics 8.4. DM-Count·P2PNet 가중치는 git clone에 포함됨. 시험 입력은 `refs/DM-Count/example_images/1.png` (DM 173명, P2PNet 머리 145개, 최대 ㎡당 0.7명 정상).
- 머리 데이터 준비 완료 (2026-11-11): CrowdHuman val(4,370장)만 받아 `src/convert_crowdhuman.py`로 변환 → `data/head_yolo/` (처음엔 val만으로 시험 → 이후 train 01~03까지 받아 최종 train 15,000장 · val 4,370장). mask·ignore 영역은 회색으로 칠해 지움. 머리의 30%가 16px 미만 → P2 탐지층 근거.
- CPU 학습 시험 통과: 200장·1 epoch·imgsz 640에 51초 (i7-13700KF). 결과는 `runs/detect/runs/<name>/` (Ultralytics가 경로를 한 번 더 감쌈). mAP≈0은 1 epoch라 정상.
- `models/head_p2.yaml` 작성·검증: Ultralytics에 YOLO11용 P2 구조가 없어(v8용만 있음) 직접 수정. COCO 가중치 297/593 이식, 2.67M 파라미터, 10.5 GFLOPs(원본 6.7), 탐지 stride 4/8/16/32.
- 학습은 Colab 대신 이 PC GPU(RTX 4070)로 함. train 15,000 / val 4,370, imgsz 960, 30 epoch, AdamW lr 0.002.
  - P2는 batch 16에서 GPU 메모리 부족으로 죽어 batch 8로 학습(nbs=64라 유효 batch 같음). 22 epoch에서 멈췄다가 resume.
- 학습·평가 완료 (2026-11-12). 보고서 `results/head_detection_report.pdf` (6쪽).
  | 모델 | mAP50 | mAP50-95 | 인원 MAE | 찾은 비율 | 61명+ 사진 | ms/장 |
  |---|---|---|---|---|---|---|
  | 전신 YOLO (X) | - | - | 8.15 | 62.0% | 21.3% | 4.9 |
  | head_base | 0.812 | 0.504 | 3.10 | 87.0% | 79.4% | 4.8 |
  | head_p2 | 0.828 | 0.516 | 3.40 | 84.9% | 77.2% | 6.8 |
  - 핵심: 붐빌수록 전신은 못 셈(101%→21%), 머리는 유지(95%→79%). 인원 오차 62% 감소.
  - P2: mAP는 올랐지만(박스 위치 정확도) 16px 미만 머리 재현율은 base 48.7% / P2 48.4%로 그대로, 속도는 느려짐 → **인원 세기에는 head_base 사용**. "탐지층 추가만으로 작은 머리 문제는 안 풀림"을 실험 결과로 발표.
- `src/crowd_alarm.py` 완성 (2026-11-14): P2PNet 대신 head_base, 키·화각 가정 대신 바닥 4점 호모그래피.
  - 발 위치 = 머리 박스 아래로 머리 높이×6 (원근 변환은 바닥 평면에만 맞으므로). 바로 위 카메라면 `--foot 0`.
  - 경보: 단계별로 '그 단계 이상' 시작 시각을 기억 → 5초 이상 이어진 가장 높은 단계. 오르내림으로 인한 헛경보 방지. 경보 시작·해제를 `<출력>.csv`에 기록.
  - 1.png(눈높이 사진, 바닥 12m×20m 가정)에서 머리 123명, 최대 ㎡당 1.75명 정상. 경보 로직은 가짜 밀도 열로 시험(8초 주의, 17초 경계, 20초 해제).
  - 바닥 설정은 `<입력>.floor.json`에 저장돼 다음 실행부터 재사용.
- 다음 할 일: **위에서 내려다본 시연 영상 확보**(직접 촬영 또는 공개 영상) → 작은 머리(입력 크기·타일링) 비교 → 전신·머리 결합 → ShanghaiTech 평가.

## 8. 다른 PC에서 시작하기 (GitHub에 없는 것)
> AI 도우미에게: 이 저장소를 새로 받은 PC라면 **먼저 `python src/check_setup.py`를 실행**해 빠진 항목을 사용자에게 알려 주고, 아래 방법으로 받도록 안내한다.

GitHub에는 코드·설정·보고서·평가 수치만 있다. 아래는 용량·라이선스 때문에 빠져 있다.

**빠른 시작 (가중치 백업 zip이 있을 때 → 이 PC와 같은 상태로 시연·평가 가능)**
1. GitHub Desktop → File → Clone repository → `kimhyeonjung0105/CrowdFace`
2. 팀 드라이브의 `CrowdFace_models_backup.zip`(head_base·head_p2·yolo11n.pt, 15MB)을 **바탕화면이나 다운로드 폴더**에 받기 (압축은 안 풀어도 됨)
3. `cv_project` 폴더에서 파이썬 환경 만들기: `py -3.13 -m venv .venv` → `.venv\Scripts\activate` → `python -m pip install -r requirements.txt`
4. `python src/check_setup.py` → zip을 찾아 `.pt`를 `models/`로 **자동 복원**하고 남은 빠진 항목을 알려 줌 (zip이 다른 곳에 있으면 `python src/check_setup.py <zip경로>`)
5. 기존 시연 코드까지 쓰려면 아래 표의 `refs/` 두 줄(git clone). 학습을 다시 할 때만 CrowdHuman 데이터 필요.

| 빠진 것 | 언제 필요 | 받는 방법 |
|---|---|---|
| `.venv` (파이썬 환경) | 항상 | `py -3.13 -m venv .venv` → `.venv\Scripts\activate` → `pip install -r requirements.txt` |
| GPU용 torch | 학습할 때 | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132` |
| `refs/DM-Count` | 기존 시연(`crowd_*.py`) | `git clone https://github.com/cvlab-stonybrook/DM-Count refs/DM-Count` (가중치 포함) |
| `refs/CrowdCounting-P2PNet` | 기존 시연(`crowd_heads.py`, `crowd_final.py`) | `git clone https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet refs/CrowdCounting-P2PNet` (가중치 포함) |
| `models/yolo11n.pt` | 항상 (전신 X, 학습 시작점) | `python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"` → 생긴 파일을 `models/`로 |
| `models/head_base.pt`, `head_p2.pt` | 머리 탐지·평가 | **백업 zip + `check_setup.py`로 자동 복원** (위 빠른 시작). 없으면 아래 데이터 준비 후 `python src/train_head.py base 30 960 16` / `p2 30 960 8` |
| `data/crowdhuman/` | 학습·평가 | 아래 다운로드 (약 11GB) |
| `data/head_yolo/` | 학습·평가 | `python src\convert_crowdhuman.py data\crowdhuman\annotation_train.odgt data\crowdhuman\Images train` 그리고 같은 명령의 `val` 버전 |
| `runs/head/*/weights` | 없어도 됨 | 곡선용 `results.csv`는 GitHub에 있음 |

CrowdHuman 다운로드 (Hugging Face `sshao0516/CrowdHuman`, CC BY-NC 4.0 — **받은 파일은 절대 GitHub에 올리지 않는다**):
```bash
cd data/crowdhuman
U=https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main
for f in annotation_train.odgt annotation_val.odgt CrowdHuman_val.zip CrowdHuman_train01.zip CrowdHuman_train02.zip CrowdHuman_train03.zip; do curl -L -O $U/$f; done
for f in CrowdHuman_*.zip; do unzip -q -o $f; done     # 모두 Images/ 로 풀림 (19,370장)
```
- 다 받은 뒤 `python src/check_setup.py`를 다시 실행해 모두 [OK]인지 확인.
- 평가 다시 하기: `python src\eval_heads.py` → `python src\bench_speed.py` → `python src\make_report.py`.

