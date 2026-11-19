# CrowdFace: 군중 밀집도 기반 인파 사고 조기 경보 — 프로젝트 안내

> 이 파일은 VS Code에서 작업할 때(사람·AI 도우미 모두) 먼저 읽는 안내서입니다.
> 답변과 주석은 **한국어**로, 짧은 목록 위주로 씁니다.

## 1. 과제 개요
- 프로젝트 이름 **CrowdFace** (GitHub 저장소 이름도 같게). 학습 데이터셋 CrowdHuman과 헷갈리지 말 것.
- 강남대학교 컴퓨터비전 미니프로젝트 (202104183 김현중, 202204201 김태섭)
- 일정: 5주차(10/16) 계획서(주제) 발표 완료 → **중간 발표(미니프로젝트 #2, 2026-11 공지)** → **14주차 최종 발표·시연(12월 셋째 주쯤)**
  - claude.ai 페이지(만든 계정 전용, 다른 계정은 Share로 공유받아야 열림): 중간 발표 자료 https://claude.ai/artifact/SEBzJkU6KdMkXfo6nrbhBD · YOLO11 학습 과정 해설 https://claude.ai/artifact/QupsSEAb49zHyrgRCFdjTx · YOLO11 학습 전체 코드(원문) https://claude.ai/artifact/8WeFp7nZ3n76UyVwmcG5CD (같은 내용 docs/YOLO11_학습_전체코드.md) · 최종 발표 초안 https://claude.ai/artifact/UBK654XpUndFrCvqPewSxF
  - 중간 발표 평가 항목: 이미지 전처리 설명 · 1차 참조 코드 실행 상황 · 구현 진척도(핵심 기능 동작) · 기술적 이해도 · 향후 계획. 발표 자료는 claude.ai 아티팩트 'CrowdFace 중간 발표'(10장·7~8분, 발표자 노트 포함).
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
src/crowd_alarm.py    ★ 최종 시연: 우리 머리 모델 + 바닥 4점 원근 변환 + 2m 칸 + 5초 경보 (사진·영상·프레임 폴더, --floor/--size 또는 클릭, --levels 시연 기준)
src/convert_ht21.py   HT21(CroHD) 머리 정답 → YOLO 라벨 (01·04 + 02 앞 80% 학습, 02 뒤 20% 검증, 03은 시험용이라 제외)
src/calib_conf.py     인원 세기용 신뢰도 기준 찾기 (검증 데이터에서 센 인원 합 = 실제 인원 합)
src/eval_video.py     HT21 영상 평가: 전신 vs 머리 인원·㎡당 밀도·흐름 일치·경보 단계 → results/eval_video/
src/tile_detect.py    타일링 탐지: 사진을 n×n 조각내 확대 탐지 + 잘린 박스 버리기 + 겹친 박스 합치기(NMS 직접 구현) (in.jpg out.jpg [격자 모델 conf])
src/eval_tiles.py     타일링 평가: 보통 vs 2×2 vs 3×3, 방식마다 신뢰도 기준을 따로 맞춘 뒤 비교 (crowdhuman | ht21) → results/eval_tiles/
src/fuse_count.py     전신+머리 결합 세기: 머리 중심이 전신 박스 위쪽 1/3 안이면 같은 사람, 가까운 짝부터 1:1 (in.jpg out.jpg [모델 머리conf 전신conf])
src/eval_fuse.py      결합 평가: 전신만 vs 머리만 vs 결합 (crowdhuman | ht21) → results/eval_fuse/
src/homography.py     원근 변환 직접 구현(점 4쌍 → 8원 연립방정식). 실행하면 OpenCV와 같은 답인지 확인. crowd_alarm.py의 Floor가 이걸 씀
src/eval_dense.py     초밀집 평가(JHU-CROWD++ 시험 1,600장): 머리(보통·타일링) vs P2PNet vs DM-Count vs 전신, 인원 구간별·빽빽함×머리 크기별 재현율 → results/eval_dense/
src/hybrid_count.py   머리 탐지 + 밀도 지도(DM-Count) 결합: 1판 칸마다 고르기(빽빽함 c) · 2판 원근 머리 크기(실패) · 3판 비율 다시 나누기 · head_size_line(화면 높이별 머리 크기 직선)
src/eval_hybrid.py    결합 평가(JHU 검증 500장에서 기준 → 시험 1,600장): 인원 오차·GAME (기본 / game / v2)
src/eval_floor_hybrid.py  HT21-03 바닥 2m 칸 밀도·경보: 탐지만 vs 밀도 지도+머리 크기 직선으로 발 위치 vs 1판
src/cvio.py           한글 경로에서도 되는 imread/imwrite (Windows OpenCV는 한글 경로를 못 읽음)
src/compare.py        YOLO vs DM-Count 비교 → results/cmp_*.jpg
src/convert_crowdhuman.py  CrowdHuman .odgt → YOLO 머리 라벨 (data/head_yolo/)
models/head_p2.yaml   직접 수정한 구조: YOLO11n + P2 탐지층, nc=1
notebooks/train_head_colab.ipynb  Colab 학습 (예비용, 실제 학습은 이 PC GPU)
src/train_head.py     이 PC GPU 학습 (base|p2 [epochs imgsz batch] / resume) → models/head_<base|p2>.pt
src/eval_heads.py     전신 vs 머리 인원 오차·머리 크기별 재현율 (CrowdHuman val) → results/eval/ (EVAL_CONF=0.15면 results/eval_conf015/)
src/bench_speed.py    세 모델 번갈아 속도 재측정 → summary.json speed_ms 덮어씀 (eval 뒤에 실행)
src/make_report.py    학습 결과 PDF → results/head_detection_report.pdf
src/check_setup.py    다른 PC에서 clone 후 빠진 것 점검 (환경·refs·가중치·데이터) → 8절
docs/YOLO11_학습_전체코드.md      학습 경로 원문 + 한 줄씩 주석 (8.4.208)
docs/YOLO11_사람탐지_원본코드.md  기존 X 예측 경로 원문 + 주석 (팀원 공부 자료, 8.4.211, PDF에 핵심 표시)
docs/YOLO11_모델구조_설명.md      backbone·넥·Detect 그림 설명 (그림 docs/figs/)
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
- **인원 세기용 신뢰도 기준 0.15** (2026-11-14, `calib_conf.py`): Ultralytics 기본 0.25는 mAP용이라 덜 셈(87%).
  CrowdHuman 검증에서 0.15면 찾은 비율 100.4%, MAE base 3.10→2.28 · **P2 3.40→2.17 (P2가 더 정확해짐)** → 보고서 P2 결론 갱신 필요.
- **시연·평가 영상 = HT21(CroHD) 데이터셋** (MOTChallenge Head Tracking 21, CC BY-NC-SA 3.0, 위에서 내려다본 군중 + 머리 박스 정답, `data/ht21/`).
  - 시험 영상 HT21-03(시부야 스크램블 교차로, 40초, 화면당 평균 258명). 바닥 10m×28m = 횡단보도 줄 간격 0.9m 가정 + 맨홀 타원 비율 0.21로 추정 (`img1.floor.json`).
  - 정답 기준 실제 밀도는 ㎡당 최대 2.6명(1초 평균) → **서울시 기준으로는 40초 내내 정상**(모든 모델도 정상, 100% 일치).
  - 결과 (`results/eval_video/HT21-03.json`, 그래프 `_count.png`·`_density.png`):
    | 모델 | 찾은 비율 | 인원 MAE | 밀도 흐름 일치(상관) |
    |---|---|---|---|
    | 전신 YOLO 960 / 1920 | 6% / 27% | 241 / 187 | **-0.38 / -0.51** (붐빌수록 오히려 낮게 봄) |
    | head_base / head_p2 | 72% / 72% | 72 / 72 | 0.79 / 0.72 |
    | head_ht21 (현장 추가 학습) | **77%** | **60** | 0.75 |
  - 머리 모델은 처음 보는 환경에서 뒤돌아선 검은 머리+검은 외투를 놓침 → 밀도를 실제의 약 0.65배로 봄 → **놓치면 위험을 낮게 본다**(발표 포인트).
- `head_ht21.pt` (2026-11-14): head_base에서 시작해 HT21-01·02·04로 20 epoch 추가 학습 (batch 8, 5분). HT21-02 검증 mAP50 0.70→0.82. 신뢰도 기준 0.20(HT21-02 검증에서 102%).
  - batch 16은 한 장에 머리 200개라 GPU 메모리가 넘쳐 5.6초/회로 느려짐 → batch 8은 8.5회/초.
- 경보 보완: 칸별 밀도를 **1초 평균**한 뒤 단계 판단 (2m 칸은 1명 차이 = ㎡당 0.25명이라 프레임마다 출렁여 '5초 연속'이 계속 초기화됨).
- 시연 모드 `--levels 1,1.33,1.67` (서울시 기준의 1/3, 화면에 '시연용 기준' 표시). 정답 경보 5초 주의→12초 경계→17초 심각→…→39초 해제, head_ht21은 16초 주의→29초 해제.
  - 시연 영상: `python src/crowd_alarm.py data/ht21/HT21/train/HT21-03/img1 results/demo_ht21_03.mp4 --model head_ht21 --conf 0.2 --levels 1,1.33,1.67` (80초, GitHub 제외)
- 폴더가 한글 경로로 옮겨지며 `cv2.imread/imwrite`가 조용히 실패 → `src/cvio.py`로 교체 (영상 입출력은 한글 경로도 됨).
- 보고서 PDF 갱신 (2026-11-14, 10쪽): 4-4 신뢰도 기준(0.15에서 전신 MAE 6.79·74.7%, 61명+ 30.3% / base 2.28·98.1% / P2 2.16·99.7%) + 5절 HT21 영상 평가·경보 비교. 다시 만들기: eval_heads → EVAL_CONF=0.15 eval_heads → eval_video → make_report.
- **타일링 실험** (2026-11-14, `tile_detect.py`·`eval_tiles.py`, head_base, 방식마다 '센 합 = 실제 합' 기준을 따로 맞춤):
  | CrowdHuman 검증 | 기준 | MAE | 재현율 | 16px 미만 재현율 | ms/장(사진 읽기 포함) |
  |---|---|---|---|---|---|
  | 보통 | 0.15 | 2.28 | 79.2% | 54.8% | 22 |
  | 타일 2×2 | 0.25 | **1.99** | 83.6% | **63.4%** | 59 |
  | 타일 3×3 | 0.27 | 2.19 | 84.1% | 64.2% | 86 |
  - P2 층으로 안 풀린 작은 머리가 **입력을 키우니 풀림**(+8.6%p) → "탐지층이 아니라 입력 해상도가 원인" 가설 확인. 2×2면 충분, 3×3은 느리기만 함.
  - **HT21-03에서는 효과 없음**: 찾은 비율 head_base 72→70→67%, head_ht21 73→74→67%. 이 영상의 머리는 높이 26~45px(중앙 33px)로 작지 않음 → 놓치는 원인은 크기가 아니라 모양(검은 뒤통수). 3×3은 너무 확대돼 헛박스가 늘어남(conf 0.2에서 정밀도 81%→49%).
  - 결론: 타일링은 머리가 작게 찍히는 먼 카메라용. 시연 영상(HT21-03)은 보통 탐지 그대로.
  - 참고: head_ht21 기준을 0.01 단위로 다시 찾으면 0.22(찾은 비율 73.3%). 보고서·eval_video는 0.05 단위로 찾은 0.20(76.7%).
- 보고서 4-5절에 타일링 반영 완료.
- **전신+머리 결합 실험** (2026-11-14, `fuse_count.py`·`eval_fuse.py`): **효과 없음.**
  - CrowdHuman 검증(기준 다시 맞춤): 머리만 MAE 2.28·재현율 79.2% → 결합(전신 0.7) MAE 2.39·79.1%. 전신 기준을 낮출수록 더 나빠짐(0.5: 2.56, 0.3: 2.96).
  - 이유: 전신이 찾는 사람은 머리 모델도 이미 찾음. 300장에서 '전신만' 박스 202개 중 진짜 새 사람 18개(9%), 중복 45개, 위쪽에 정답 머리 없음 139개.
  - HT21-03: 전신(입력 1920)이 추가한 사람은 1,000프레임 합쳐 219명 → 변화 없음(73.3%→73.4%).
  - HT21-03에서 박스 위치까지 맞힌 재현율은 head_ht21 60.7%·정밀도 82.8%(head_base 53.7%·74.4%) → '찾은 비율 73%'에는 헛박스가 섞여 있음.
  - 발표 결론: "X+Y 결합"이 아니라 **"붐비면 Y(머리)가 X(전신)를 대신한다"**. 3절의 '전신·머리 연결 규칙'은 구현·검증했지만 인원 세기에는 쓰지 않음.
- **호모그래피 직접 구현** (`homography.py`): 점 4쌍 → 식 8개 연립방정식(np.linalg.solve). OpenCV와 행렬 차이 1e-14, 점 변환 차이 0.002mm. `crowd_alarm.py`의 Floor가 이 함수를 쓰도록 교체(구역 안 인원이 예전과 다른 프레임 1/1000, 경계 위 점의 반올림 차이).
- 보고서 4-6절(결합)·5절(호모그래피 직접 구현) 반영 완료 (11쪽).
- **방향 결정 (2026-11-17)**: 주제 유지, 난이도 보강. A 초밀집 검증 → A' 탐지+밀도 지도 결합 → B 머리 추적으로 속도·정체 → C 기존 방법 비교 → D 직접 촬영.
- **A. 초밀집 검증** (2026-11-17, `eval_dense.py`, JHU-CROWD++ v2.0 시험 1,600장, 사진당 중앙값 118명·최대 9,059명, 머리 폭 중앙값 11px. JHU로 학습·기준 정하기 안 함):
  | 사진당 인원 | 0-50 | 51-200 | 201-500 | 501-1000 | 1001-3000 | 3001+ |
  |---|---|---|---|---|---|---|
  | 머리 보통 (센 비율 / MAE) | 89% / 6 | 82% / 31 | 70% / 111 | 55% / 327 | 31% / 1075 | 6% / 4611 |
  | 머리 타일 2×2 | 87% / 6 | 80% / 30 | 71% / 104 | 60% / 288 | 44% / 884 | 12% / 4330 |
  | P2PNet (기존, 점) | 184% / 30 | 129% / 49 | 118% / 95 | 112% / 194 | 107% / 376 | 86% / 1204 |
  | DM-Count (기존, 밀도 지도) | 166% / 20 | 133% / 42 | 119% / 80 | 109% / 149 | 99% / 301 | 79% / 1208 |
  | 전신 YOLO | 50% / 13 | 12% / 97 | 3% / 313 | 1% / 678 | 0% | 0% |
  - **머리 탐지는 초밀집에서 무너진다.** 200명까지는 머리 탐지가 가장 정확(MAE 6·31 vs 밀도 지도 20·42), 500명 넘으면 밀도 지도가 훨씬 정확. 밀도 지도는 한산한 사진을 30~80% 부풀림 → **둘을 섞을 근거**(A').
  - 원인 두 가지(같은 머리 크기 안에서 비교): ① 작은 머리 — 8px 미만 재현율 1~21%, 8~16px 23~47% ② **머리가 겹칠 만큼 빽빽함**(빽빽함 c = 이웃 머리 3개 평균 거리 ÷ 머리 폭 < 1) — 32px 이상 머리도 c<1이면 45%(c 2~3이면 78~80%). P2PNet은 c<1에서도 68%로 덜 무너짐.
  - ㎡ 환산(근사, HT21-03 카메라 기준): 사진 속 머리 폭 1개 ≈ 바닥 0.53m → **c≈1.0 ↔ ㎡당 4명, c≈0.9 ↔ 5명**. 즉 이 각도의 카메라에서는 경계·심각 단계가 정확히 탐지가 무너지는 구간. 위에서 수직으로 보면 같은 밀도에서 c가 훨씬 커서(머리 실제 폭 기준) 덜 무너짐 → 카메라는 높고 수직일수록 유리.
- **A'. 탐지 + 밀도 지도 결합** (2026-11-17, 기준은 JHU 검증 500장, 평가는 시험 1,600장):
  | 방법 | 인원 MAE | GAME1 | GAME2 | GAME3(64칸) | 0-50명 | 51-200 | 201-500 | 501-1000 | 1001+ |
  |---|---|---|---|---|---|---|---|---|---|
  | 머리 탐지만 | 191.5 | 196.5 | 202.6 | 212.7 | **6** | 31 | 111 | 327 | 1075~4611 |
  | 밀도 지도만 | 84.2 | 99.0 | 111.3 | 128.5 | 20 | 42 | 80 | 149 | **301~1208** |
  | **1판 칸마다 고르기** (블록 64px, c<1.5면 밀도 지도) | **76.8** | **92.5** | **107.1** | **125.8** | 14 | **30** | **64** | **133** | 333~1313 |
  - 1판 약점: 밀도 지도의 헛것(조명·관중석 무늬: 실제 42명→982명, 사람 없는 사진→302명; 0~50명 구간 오차의 47%가 12장)을 못 거름.
  - **2판(원근 직선으로 그 자리 머리 크기 예측 → 헛것 거르기) 실패**: 검증 75.9·시험 83.6(1판보다 나쁨). 3판(구역별 밀도 지도 인원 ÷ 탐지 수로 탐지에 무게) JHU GAME3 136.7로 실패.
  - **HT21-03(시연 영상)에서는 밀도 지도가 훨씬 정확**: 찾은 비율 탐지 77% vs 밀도 지도 106%, 인원 오차 60 vs 18.5. 1판은 밀도 지도를 4% 칸만 써서 효과 없음(탐지가 놓치는 원인이 빽빽함이 아니라 검은 뒤통수라 규칙에 안 걸림).
  - **바닥 2m 칸·경보 비교 (`eval_floor_hybrid.py`, 시연용 기준)**:
    | 방법 | 구역 인원 | 최대 칸 밀도(정답 1.62) | 흐름 일치 | 단계 일치 | 경보 |
    |---|---|---|---|---|---|
    | 머리 탐지만(현재) | 0.75 | 1.13 | 0.71 | 17% | 16초 주의 → 29초 해제 |
    | **밀도 지도 + 머리 크기 직선으로 발 위치** | **0.98** | **1.45** | **0.82** | **54%** | 7초 주의→14초 경계→**25초 심각**→36초 해제 (정답 5초 주의→12초 경계→17초 심각→39초 해제) |
    | 1판 | 0.73 | 1.08 | 0.76 | 12% | 20초 주의 → 28초 해제 |
  - 해석: 인원은 밀도 지도(기존 방법)가, **발 위치에 필요한 머리 크기는 우리 머리 탐지가** 맡는 역할 분담이 경보에 가장 좋음. 한산한 장면(200명 이하)은 머리 탐지가 더 정확(JHU 0~50명 MAE 6 vs 20).
- **DM-Count 혼자도 비슷함** (2026-11-17): 바닥 칸 경보에서 DM-Count만 쓰고 머리 크기를 사람이 33px로 넣으면 단계 일치 70%(머리 YOLO 직선 54%, 20px 45%, 50px 56%, 발 위치 없이 17%). → 결합에서 머리 YOLO의 역할은 '설치 때 머리 크기 자동 측정' 정도라 "DM-Count로 센 것"과 차이가 작음. 최종 시스템은 **머리 YOLO 중심 유지** 결정.
- **발 위치 가정 확인 (2026-11-18, `check_foot.py`)**: '머리 높이 × 6'은 확인 안 한 가정이었음(정답 밀도에도 같은 가정을 써서 모델 비교는 공정하지만 절대 위치는 미검증). 시연 영상에서 전신 YOLO 박스 발과 짝지은 1,167명으로 재니 **× 4.79**(화면 높이와 무관). ×6이면 바닥에서 1.49m 어긋남(2m 칸 거의 한 칸) → ×4.79면 0.67m. → **바닥 밀도는 '보완 중'**. → **수정 완료**: `foot_calib.py`로 전신(X)과 짝지어 박스 출처마다 자동 보정(정답 ×4.79, head_base ×5.42, head_p2 ×5.51, head_ht21 ×5.47 — 모델 박스가 정답보다 작아 배수가 큼). `crowd_alarm.py --foot auto`(기본, floor.json에 모델별 저장).
  - 다시 돌린 HT21-03: 밀도 흐름 일치 head_ht21 0.75→**0.87**, head_base 0.79→0.82, 전신 -0.41/-0.51. 정답 최대 ㎡당 2.64명. 시연 기준 정답 경보 8.6초 주의→12초 경계→18.3초 심각→26.8 경계→28.6 주의→33초 해제, head_ht21 18초 주의→28.4초 해제. 머리 모델은 여전히 밀도를 약 0.7배로 봄.
  - 바닥 칸 비교(eval_floor_hybrid): 탐지만 단계 일치 22%·흐름 0.87, 밀도 지도+머리 크기 직선 55%·0.83, 1판 19%. 초밀집 ㎡ 환산: c≈1.0 ↔ ㎡당 3.9명, 5명 ↔ c≈0.88.
- **검증 대기 (2026-11-18)**: 바닥 원근 변환 · 발 위치 자동 보정 · 5초 경보와 시연 영상은 동작하지만 팀원이 직접 검증하기 전까지 '보완 중'으로 표시(중간 발표 2장). 검증 후 다음 발표에서 완료로 올림.
- **2026-11-18 이어서 할 것**: ① 바닥 원근 변환·발 위치 자동 보정·5초 경보는 팀원이 직접 검증 후 '완료'로 ② 무시 영역 처리 3가지(라벨만 빼기 / 회색 + 절반 넘게 덮인 라벨 빼기 / 손실에서 빼기) 비교 학습 — 손실에서 빼는 곳은 utils/loss.py v8DetectionLoss의 bce_loss (해설 문서 9절) ③ 회색이 절반 넘게 덮은 머리 라벨 약 1,000개(0.35%) 빼기 ④ head_ht21은 사람 없는 배경에서 헛탐지 많음(회색 실험에서 185개) — 추가 학습 때 확인.
- **다음 목표 (중간 발표 이후)**: 머리 모델 추가 학습으로 DM-Count 수준 따라잡기 — 시연 영상 찾은 비율 77% → 100% 안팎(DM-Count 106%). 위에서 본 군중·어두운 뒤통수 데이터, 학습량·입력 크기 늘리기. 그다음 결합(칸마다 고르기)을 영상 경보에 적용 → B 움직임 → D 직접 촬영.

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

HT21 시연·평가 영상 (MOTChallenge, CC BY-NC-SA 3.0 — GitHub에 올리지 않음, 약 4.1GB):
```bash
mkdir -p data/ht21 && cd data/ht21 && curl -L -O https://motchallenge.net/data/HT21.zip
python -c "import zipfile; z=zipfile.ZipFile('HT21.zip'); z.extractall('.', [n for n in z.namelist() if n.startswith('HT21/train/')])"
```
- Git Bash의 unzip은 4GB 넘는 zip을 못 풀어서 파이썬으로 푼다. 바닥 설정 `HT21/train/HT21-03/img1.floor.json`은 CLAUDE.md 7절 값으로 다시 만든다:
  `{"floor": [[462,200],[1458,200],[1773.5,1010],[146.5,1010]], "size": [10,28]}`
- 추가 학습 데이터: `python src/convert_ht21.py` → `python src/train_head.py ht21 20 960 8`

JHU-CROWD++ 초밀집 평가용 (학술·비상업 전용, 3.1GB — GitHub에 올리지 않음):
```bash
pip install gdown && mkdir -p data/jhu && cd data/jhu && gdown --folder "https://drive.google.com/drive/folders/1FkdvHyAom1B2aVj6_jZpZPW01sQNiI7n"
python -c "import zipfile; zipfile.ZipFile('jhucrowdv2.0/jhu_crowd_v2.0.zip').extractall('.')"     # → data/jhu/jhu_crowd_v2.0/{train,val,test}
```

CrowdHuman 다운로드 (Hugging Face `sshao0516/CrowdHuman`, CC BY-NC 4.0 — **받은 파일은 절대 GitHub에 올리지 않는다**):
```bash
cd data/crowdhuman
U=https://huggingface.co/datasets/sshao0516/CrowdHuman/resolve/main
for f in annotation_train.odgt annotation_val.odgt CrowdHuman_val.zip CrowdHuman_train01.zip CrowdHuman_train02.zip CrowdHuman_train03.zip; do curl -L -O $U/$f; done
for f in CrowdHuman_*.zip; do unzip -q -o $f; done     # 모두 Images/ 로 풀림 (19,370장)
```
- 다 받은 뒤 `python src/check_setup.py`를 다시 실행해 모두 [OK]인지 확인.
- 평가 다시 하기: `python src\eval_heads.py` → `python src\bench_speed.py` → `python src\make_report.py`.

