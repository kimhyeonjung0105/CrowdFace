# YOLO11 사람 탐지 — 원본 코드 따라가기 (기존 기능 X)

> 공부용 정리본. 코드는 Ultralytics **8.4.211** 원본(`.venv/Lib/site-packages/ultralytics/`)에서 그대로 가져왔고, 설명용 docstring만 뺐다.
>
> 버전 주의: 이 문서는 8.4.211 기준, 학습 PC와 `docs/YOLO11_학습_전체코드.md`는 **8.4.208** 기준이라 원본 줄 번호가 조금 다를 수 있다. 내용(구조·계산)은 같다.
> 주석(`#`, 초록색)은 우리가 단 한국어 설명이며 **설명하는 코드 줄 바로 위**에 있다.
>
> 기준: COCO로 학습된 `yolo11n.pt`(80종류) + `src/yolo_person.py` 설정(입력 640, 신뢰도 0.25, 사람만).
> 예시 숫자: `refs/DM-Count/example_images/1.png` (1695×1271)을 실제로 돌려서 잰 값.

**읽는 법** — <span style="color:#b3261e;font-weight:700">빨간 네모</span> = 꼭 이해할 핵심 · 흐린 코드 + 오른쪽 괄호 = 덜 중요한 부분 (주석은 있으니 한 번씩 훑어보기)

---

## 0. 전체 흐름 한눈에

| 단계 | 하는 일 | 1.png 실제 값 | 파일 · 함수 |
|---|---|---|---|
| ① 실행 스크립트 | 모델 불러서 사진 넣기 | `classes=[0]` = 사람만 | `src/yolo_person.py` |
| ② 전처리 | 비율 유지하며 줄이고 회색으로 채우기, BGR→RGB, ÷255 | 1695×1271 → **640×480** | `predictor.py` `pre_transform` · `preprocess`, `augment.py` `LetterBox` |
| ③ 모델 실행 | yaml 순서대로 층 23개 통과 | 특징 지도 80×60 · 40×30 · 20×15 | `tasks.py` `_predict_once` |
| ④ Detect 층 | 칸마다 박스 4개 숫자 + 80종류 점수 | **6,300칸 × 84개** | `head.py` `Detect` |
| ⑤ 박스 풀기 | DFL로 거리 → 박스 좌표 | | `block.py` `DFL`, `tal.py` `make_anchors` · `dist2bbox` |
| ⑥ 후처리 | 점수 0.25 넘는 '사람' 후보 → 겹친 것 합치기(NMS) | 후보 **53개 → 7명** | `nms.py` `non_max_suppression` |
| ⑦ 좌표 되돌리기 | 640×480 기준 박스를 원본 1695×1271로 | | `predict.py` `construct_result`, `ops.py` `scale_boxes` |

**칸 수 계산**: 640×480 입력 → stride 8·16·32 지도 = 80×60 + 40×30 + 20×15 = 4,800 + 1,200 + 300 = **6,300칸**

**칸당 출력**: 박스 4개(x, y, w, h) + COCO 80종류 점수 = **84개** (내부적으로는 박스 64 + 클래스 80 = 144)

> 위 숫자는 1.png 기준이다. 다른 사진을 넣으면 크기·칸 수·사람 수는 달라지지만, 코드가 지나가는 순서(②~⑦)는 똑같다.

---

## 1. 실행 스크립트 (팀 작성, YOLO 호출만): `src/yolo_person.py`

> 이 파일만 팀이 쓴 코드다. 사진을 넣고 Ultralytics YOLO를 부른 뒤 결과를 그릴 뿐, YOLO 자체는 건드리지 않는다.
> **2~7절은 전부 Ultralytics 원본 코드**이고, 주석의 "우리"는 설명을 위한 표현일 뿐 우리가 고친 곳이 아니다.

```python
#@dim 설명문 + 준비: 경로·라이브러리 불러오기
"""기존 기능 X: COCO 사전학습 YOLO로 사람 '전신' 박스를 찾아 인원 세기 (코드 수정 없음).
머리 탐지(추가 기능 Y)와 비교하기 위한 기준선.
usage: python src/yolo_person.py in.jpg out.jpg [imgsz=640]
VS Code ▶ 버튼으로 인자 없이 실행하면 아래 기본값을 씀."""
# sys = 명령줄 인자 읽기, pathlib = 경로 다루기
import sys, pathlib as _pl
# ROOT = cv_project 폴더 (이 파일 src/yolo_person.py의 두 단계 위)
ROOT = _pl.Path(__file__).resolve().parent.parent
# 가중치 폴더 = cv_project/models
MODELS = str(ROOT / "models")
# OpenCV: 그림 그리기용
import cv2
# 팀이 만든 도우미: 한글 경로에서도 되는 사진 읽기·쓰기 (Windows OpenCV는 한글 경로를 못 씀)
from cvio import imread, imwrite
# Ultralytics YOLO
from ultralytics import YOLO

#@key 기본값: 인자 없이 실행하면 쓰는 값 (사진에 따라 자동으로 바뀌지 않음)
# 입력 사진 = DM-Count 저장소에 들어 있는 시험용 군중 사진 (1695×1271)
SRC = str(ROOT / "refs/DM-Count/example_images/1.png")
# 결과 사진 저장 위치
OUT = str(ROOT / "results/yolo_person.jpg")
# 입력 크기 640 = Ultralytics 기본값. 붐비는 사진은 1920이면 훨씬 많이 찾음 (1.png: 7명 → 45명)
IMGSZ = 640

#@dim 명령줄 인자 받기 (넣으면 그 값, 안 넣으면 위 기본값)
# 예: python src/yolo_person.py 내사진.jpg 결과.jpg 1920
# 1번째 인자가 있으면 입력 사진, 없으면 기본값 SRC
src = sys.argv[1] if len(sys.argv) > 1 else SRC
# 2번째 인자가 있으면 결과 사진 위치, 없으면 기본값 OUT
out = sys.argv[2] if len(sys.argv) > 2 else OUT
# 3번째 인자가 있으면 입력 크기(숫자로 바꿈), 없으면 기본값 640
imgsz = int(sys.argv[3]) if len(sys.argv) > 3 else IMGSZ

#@key ★ 실제 탐지 — 2~7절이 전부 이 predict 한 줄 안에서 일어남
# COCO로 학습된 yolo11n.pt를 불러서 예측. 학습은 하지 않는다 (기존 기능 X는 '가져다 쓰기')
#   classes=[0] : COCO 80종류 중 0번 = person 만 남김
#   conf=0.25   : 점수 0.25 넘는 박스만
#   imgsz=640   : 긴 변을 640으로 줄여서 넣음
#   [0]         : 사진 1장만 넣었으니 결과 목록의 첫 번째
r = YOLO(MODELS + "/yolo11n.pt").predict(src, classes=[0], conf=0.25, imgsz=imgsz, verbose=False)[0]
# 남은 박스 개수 = 사람 수
n = len(r.boxes)

#@dim 결과 그림: 박스와 "YOLO person: 7" 글자를 그려 저장
# 원본 사진 위에 박스를 그린 그림 (라벨·점수 글자는 빼고 선 두께 2)
img = r.plot(labels=False, conf=False, line_width=2)
# 왼쪽 위에 검은 사각형 (글자 배경)
cv2.rectangle(img, (0, 0), (520, 60), (0, 0, 0), -1)
# 그 위에 흰 글자: "YOLO person: 7  (imgsz 640)"
cv2.putText(img, f"YOLO person: {n}  (imgsz {imgsz})", (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
# 결과 사진 저장
imwrite(out, img)

#@dim 터미널에 사람 수 출력, 사진 뷰어로 열기
# 터미널에 "persons 7 imgsz 640"
print("persons", n, "imgsz", imgsz)
print("결과 사진:", out)
import os
# Windows면 결과 사진을 사진 뷰어로 바로 열기
if hasattr(os, "startfile"): os.startfile(out)
```

---

## 2. 모델 구조: `cfg/models/11/yolo11.yaml`

`yolo11n.pt` 안에 이 구조가 같이 저장돼 있다. 한 줄 = 층 하나: `[어디서 받나, 몇 번 반복, 층 종류, 인자]`

```yaml
#@dim 파일 머리 (저작권·문서 주소)
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Ultralytics YOLO11 object detection model with P3/8 - P5/32 outputs
# Model docs: https://docs.ultralytics.com/models/yolo11
# Task docs: https://docs.ultralytics.com/tasks/detect

# Parameters
#@key 클래스 수 80 (사람은 0번)
# COCO 80종류 (사람, 자전거, 차, …)
nc: 80

#@dim 크기 배수 (n·s·m·l·x 다섯 가지 중 우리는 n)
# model compound scaling constants, i.e. 'model=yolo11n.yaml' will call yolo11.yaml with scale 'n'
scales:
  # [깊이 배수, 폭(채널) 배수, 최대 채널]
  # n(nano): 깊이 0.5배, 채널 0.25배 → 181층, 파라미터 약 262만 개, 6.6 GFLOPs  ← 우리가 쓰는 것
  n: [0.50, 0.25, 1024] # summary: 181 layers, 2624080 parameters, 2624064 gradients, 6.6 GFLOPs
  # s(small): 채널 0.5배 → 946만 개
  s: [0.50, 0.50, 1024] # summary: 181 layers, 9458752 parameters, 9458736 gradients, 21.7 GFLOPs
  # m(medium): 채널 1배, 최대 512 → 2,011만 개
  m: [0.50, 1.00, 512] # summary: 231 layers, 20114688 parameters, 20114672 gradients, 68.5 GFLOPs
  # l(large): 깊이 1배 → 2,537만 개
  l: [1.00, 1.00, 512] # summary: 357 layers, 25372160 parameters, 25372144 gradients, 87.6 GFLOPs
  # x(extra large): 채널 1.5배 → 5,697만 개
  x: [1.00, 1.50, 512] # summary: 357 layers, 56966176 parameters, 56966160 gradients, 196.0 GFLOPs

#@plain
# ── backbone: 사진을 점점 줄이며 특징(선·모양·질감)을 뽑는 부분 ──
backbone:
  # 0: 3×3 합성곱, 보폭 2 → 크기 1/2
  - [-1, 1, Conv, [64, 3, 2]]
  # 1: 크기 1/4
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  # 3: 크기 1/8  (P3)
  - [-1, 1, Conv, [256, 3, 2]]
  # 4: ← 나중에 넥이 끌어다 씀
  - [-1, 2, C3k2, [512, False, 0.25]]
  # 5: 크기 1/16 (P4)
  - [-1, 1, Conv, [512, 3, 2]]
  # 6: ← 나중에 넥이 끌어다 씀
  - [-1, 2, C3k2, [512, True]]
  # 7: 크기 1/32 (P5)
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  # 9: 여러 크기로 풀링해 넓은 범위를 봄
  - [-1, 1, SPPF, [1024, 5]]
  # 10: 주의(attention) 블록 — 중요한 위치에 가중치
  - [-1, 2, C2PSA, [1024]]

# ── head(넥): 작은 지도를 키워 큰 지도와 합침 ──
head:
  # 11: 1/32 → 1/16로 키움
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  # 12: 6번 층(1/16)과 이어 붙임
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  # 14: 1/16 → 1/8로 키움
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  # 15: 4번 층(1/8)과 이어 붙임
  - [[-1, 4], 1, Concat, [1]]
  # 16: ★ 1/8 지도 = 작은 사람 담당 (P3)
  - [-1, 2, C3k2, [256, False]]
  # 17: 다시 1/16으로 줄임
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  # 19: ★ 1/16 지도 = 중간 사람 담당 (P4)
  - [-1, 2, C3k2, [512, False]]
  # 20: 1/32로 줄임
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  # 22: ★ 1/32 지도 = 큰 사람 담당 (P5)
  - [-1, 2, C3k2, [1024, True]]

#@key 세 지도(16·19·22번)에서 탐지
  # 23: Detect — 박스와 80종류 점수를 냄
  - [[16, 19, 22], 1, Detect, [nc]]
```

---

## 3. 전처리: 사진 → 모델 입력

### 3-1. `engine/predictor.py` `BasePredictor.pre_transform`

```python
#@dim 넣은 사진들이 모두 같은 크기인지 확인
def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
    # 사진 크기들을 집합에 넣어 종류가 1개면 모두 같은 크기 (우리는 1장이라 True)
    same_shapes = len({x.shape for x in im}) == 1
#@key LetterBox(640) 준비 — .pt 모델이면 auto=True
    # 목표 크기 640. auto=True면 정사각형으로 채우지 않고 32의 배수까지만 채움
    letterbox = LetterBox(
        self.imgsz,
        auto=same_shapes
        and self.args.rect
        and not self.scale_fill
        and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) and self.model.format != "imx")),
        scale_fill=self.scale_fill,
        stride=self.model.stride,
    )
#@plain
    # 사진마다 LetterBox 적용
    return [letterbox(image=x) for x in im]
```

### 3-2. `data/augment.py` `LetterBox.get_params` — 크기 계산

```python
#@dim 원본 크기 읽기, 목표 크기 (640, 640)
def get_params(self, labels: dict[str, Any]) -> dict[str, Any]:
    # 원본 사진
    img = labels["img"]
    # 원본 [높이, 너비] = [1271, 1695]
    shape = img.shape[:2]
    # 목표 크기 (예측 때는 640)
    new_shape = labels.pop("rect_shape", self.new_shape)
    # 숫자 하나면 정사각형 (640, 640)으로
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

#@key 배율 → 줄인 크기 → 여백 (비율 유지)
    # 줄이는 배율: 높이·너비 중 더 많이 줄여야 하는 쪽 = min(640/1271, 640/1695) = 0.378
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not self.scaleup:
        r = min(r, 1.0)

    ratio = r, r
    # 줄인 크기 (너비, 높이) = (640, 480)  ← 가로세로 비율 유지
    new_unpad = max(round(shape[1] * r), 1), max(round(shape[0] * r), 1)
    # 채울 여백 = (640-640, 640-480) = (0, 160)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    # auto면 여백을 32의 배수 나머지만큼만 → 160 % 32 = 0 → 여백 없이 640×480 그대로
    if self.auto:
        dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)
#@dim scale_fill = 비율 무시하고 늘리기 (안 씀)
    # 비율을 무시하고 목표 크기에 꽉 차게 늘리는 방식
    elif self.scale_fill:
        # 여백 없음
        dw, dh = 0.0, 0.0
        # 줄인 크기 = 목표 크기
        new_unpad = (new_shape[1], new_shape[0])
        # 가로·세로 배율이 서로 달라짐 (사진이 찌그러짐)
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

#@dim 여백을 양쪽에 반씩 나누고 계산값 돌려주기
    # 가운데 정렬이면 여백을 위아래(좌우)에 반씩
    if self.center:
        dw /= 2
        dh /= 2

    # 위·아래 여백 픽셀 (홀수면 한쪽이 1px 더)
    top, bottom = round(dh - 0.1) if self.center else 0, round(dh + 0.1)
    # 왼쪽·오른쪽 여백 픽셀
    left, right = round(dw - 0.1) if self.center else 0, round(dw + 0.1)

    # 계산값을 돌려줌 → 다음 apply_image가 실제로 줄이고 채움
    return {
        "orig_shape": shape,
        "new_shape": new_shape,
        "ratio": ratio,
        "new_unpad": new_unpad,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }
```

### 3-3. `LetterBox.apply_image` — 실제로 줄이고 채우기

```python
#@dim 사진과 목표 크기 꺼내기
def apply_image(self, labels: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    # 원본 사진
    img = labels["img"]
    # 원본 [높이, 너비]
    shape = img.shape[:2]
    # 위 get_params가 계산한 줄일 크기 (640, 480)
    new_unpad = params["new_unpad"]

#@key 줄이기 (1695×1271 → 640×480)
    # 크기가 다르면 OpenCV로 줄임
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=self.interpolation)
#@dim 흑백 사진이면 채널 축 추가
    # [높이, 너비]만 있는 흑백이면 [높이, 너비, 1]로
    if img.ndim == 2:
        img = img[..., None]

#@key 여백을 회색 114로 채움
    h, w, c = img.shape
    top, bottom = params["top"], params["bottom"]
    left, right = params["left"], params["right"]
    # padding_value = 114. 모델은 학습 내내 114 회색을 '아무것도 없는 곳'으로 배웠다
    if c == 3:
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(self.padding_value,) * 3
        )
#@dim 3채널이 아닌 특수 사진용 (안 씀)
    # 색 채널이 3개가 아니면(적외선 등) 회색 판을 만들고 가운데에 사진을 붙임
    else:
        pad_img = np.full((h + top + bottom, w + left + right, c), fill_value=self.padding_value, dtype=img.dtype)
        pad_img[top : top + h, left : left + w] = img
        img = pad_img

#@dim 결과 사진과 크기를 labels에 담아 돌려주기
    # 줄이고 채운 사진 (640×480)
    labels["img"] = img
    # 목표 크기 기록
    labels["resized_shape"] = params["new_shape"]
    return labels
```

### 3-4. `BasePredictor.preprocess` — 숫자 형태 맞추기

```python
#@plain
def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
    # 사진(넘파이 배열)으로 넣었으면 (우리는 이쪽)
    if not isinstance(im, torch.Tensor):
#@key LetterBox → 축 순서 → BGR→RGB → ÷255
        # 위 LetterBox
        im = self.pre_transform(im)
        # 사진 1장이면 앞에 묶음 축 추가 → [1, 480, 640, 3]
        im = torch.from_numpy(im[0]).unsqueeze(0) if len(im) == 1 else torch.from_numpy(np.stack(im))
        # GPU/CPU로 보냄
        im = im.to(self.device)
        # [묶음, 높이, 너비, 색] → [묶음, 색, 높이, 너비] = [1, 3, 480, 640]  (PyTorch 순서)
        im = im.permute(0, 3, 1, 2).contiguous()
        # OpenCV는 BGR 순서라 RGB로 뒤집음
        if im.shape[1] == 3:
            im = im.flip(1)
        # 0~255 정수 → 0~1 실수. YOLO의 정규화는 이것뿐 (ImageNet 평균·표준편차 빼기 없음)
        im = (im.half() if self.model.fp16 else im.float()).div_(255)
#@dim 이미 텐서로 넣었으면 그대로 (우리는 해당 없음)
    # 이미 0~1 텐서로 넣은 경우: 장치·자료형만 맞춤
    else:
        im = im.to(self.device, torch.float16 if self.model.fp16 else torch.float32)
#@plain
    # 모델 입력 [1, 3, 480, 640]
    return im
```

---

## 4. 모델 실행: `nn/tasks.py` `BaseModel._predict_once`

```python
#@dim 출력 저장 목록 준비
def _predict_once(self, x, profile=False, embed=None):
    # y = 층 출력 저장, dt = 층별 시간, embeddings = 중간 특징 (뒤 둘은 안 씀)
    y, dt, embeddings = [], [], []
    # 중간 특징을 뽑을 층 번호 (안 쓰면 {-1})
    embed = frozenset(embed) if embed else {-1}
    max_idx = max(embed)
#@key yaml의 층 0~23을 순서대로 실행
    # y = 층마다 출력을 저장해 두는 목록 (Concat이 앞 층 출력을 다시 쓰기 때문)
    for m in self.model:
        # 바로 앞 층이 아닌 곳에서 받는 층이면 (예: Concat [-1, 6], Detect [16, 19, 22])
        if m.f != -1:
            # 저장해 둔 출력에서 꺼내옴
            x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
#@dim 층별 속도 측정 (안 씀)
        # profile=True일 때만 층마다 걸린 시간을 잼
        if profile:
            self._profile_one_layer(m, x, dt)
#@key 층 실행 + 나중에 쓸 출력 저장
        # 층 실행
        x = m(x)
        # 나중에 다시 쓰일 층(4, 6, 10, 13, 16, 19, 22 등)만 저장
        y.append(x if m.i in self.save else None)
#@dim 중간 특징 뽑기 기능 (안 씀)
        # embed에 지정한 층이면 출력을 평균 내서 벡터로 모음
        if m.i in embed:
            embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
            # 마지막 지정 층까지 왔으면 탐지 대신 특징 벡터를 돌려줌
            if m.i == max_idx:
                return torch.unbind(torch.cat(embeddings, 1), dim=0)
#@plain
    # 마지막 층(Detect)의 출력
    return x
```

Detect에 들어가는 세 지도 (1.png, 640×480 입력):

| 지도 | 층 | stride | 크기 [묶음, 채널, 높이, 너비] | 칸 수 |
|---|---|---|---|---|
| P3 (작은 사람) | 16번 | 8 | [1, 64, 60, 80] | 4,800 |
| P4 (중간) | 19번 | 16 | [1, 128, 30, 40] | 1,200 |
| P5 (큰 사람) | 22번 | 32 | [1, 256, 15, 20] | 300 |

---

## 5. Detect 층: `nn/modules/head.py` `Detect`

### 5-1. `Detect.__init__` — 갈래 두 개 만들기

```python
#@dim 함수 시작
# nc = 클래스 수 80, reg_max = DFL 칸 수 16, ch = 세 지도의 채널 수 (64, 128, 256)
def __init__(self, nc: int = 80, reg_max: int = 16, end2end: bool = False, ch: list[int] | tuple[int, ...] = ()):
    # 부모(nn.Module) 초기화
    super().__init__()
#@key 칸당 출력 수 144 = 클래스 80 + 박스 64
    # 클래스 수 = 80 (COCO)
    self.nc = nc
    # 탐지 지도 수 = 3 (P3, P4, P5)
    self.nl = len(ch)
    # 박스 한 변의 거리를 0~15칸 중 어디인지 16개 확률로 냄 (DFL)
    self.reg_max = reg_max
    # 칸당 출력 = 80 + 16×4 = 144
    self.no = nc + self.reg_max * 4
    # stride는 모델 만들 때 따로 잼 → [8, 16, 32]
    self.stride = torch.zeros(self.nl)
    # 중간 채널 수: c2(박스) = max(16, 64/4, 64) = 64,  c3(클래스) = max(64, min(80, 100)) = 80
    c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
#@key cv2 = 박스 갈래 (출력 64 = 4변 × 16칸)
    # 3×3 → 3×3 → 1×1. 지도마다 하나씩
    self.cv2 = nn.ModuleList(
        nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
    )
#@key cv3 = 클래스 갈래 (출력 80 = 종류마다 점수)
    # YOLO11은 아래 가벼운 구조(legacy=False)를 씀
    self.cv3 = (
        nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        if self.legacy
        else nn.ModuleList(
            nn.Sequential(
                # 채널별 3×3(DWConv) + 1×1
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                # 마지막 1×1: 80채널 = 80종류 점수  ← 머리 모델(Y)은 학습 때 nc=1이라 여기가 1채널이 됨 (코드 수정 아님)
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
    )
#@plain
    # 16칸 확률 → 거리 하나로 바꾸는 층 (5-6)
    self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

#@dim YOLO26용 갈래 (YOLO11은 안 만듦)
    # NMS 없는 모델(YOLO26)만 갈래를 하나 더 복사해 둠. YOLO11은 end2end=False
    if end2end:
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
```

### 5-2. `Detect.forward_head` — 세 지도의 칸을 한 줄로

```python
#@dim 함수 시작, 갈래가 없을 때 처리
# x = 세 지도, box_head = cv2, cls_head = cv3
def forward_head(
    self, x: list[torch.Tensor], box_head: nn.Module | None = None, cls_head: nn.Module | None = None
) -> dict[str, torch.Tensor]:
    # 갈래가 없으면(추론용으로 합친 모델) 빈 결과
    if box_head is None or cls_head is None:
        return {}
#@key 세 지도를 갈래에 통과시켜 6,300칸으로 이어 붙임
    # 묶음 안 사진 수 = 1
    bs = x[0].shape[0]
    # 박스: 지도마다 [1, 64, 높이×너비]로 펴서 이어 붙임 → [1, 64, 6300]
    boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
    # 점수: 지도마다 [1, 80, 높이×너비] → [1, 80, 6300]. 아직 sigmoid 전 값
    scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
    return {"boxes": boxes, "scores": scores, "feats": x}
```

### 5-3. `Detect.forward` — 예측 모드면 박스로 풀기

```python
#@dim 함수 시작
# x = 16·19·22번 층에서 받은 세 지도
def forward(
    self, x: list[torch.Tensor]
) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
#@key 두 갈래 실행
    # cv2 박스, cv3 클래스 (5-2)
    preds = self.forward_head(x, **self.one2many)
#@dim YOLO26용 (YOLO11은 건너뜀) · 학습 중이면 여기서 끝
    # YOLO26처럼 두 번째 갈래가 있으면 그것도 실행 (YOLO11은 없음)
    if getattr(self, "one2one_cv2", None) is not None:
        x_detach = [xi.detach() for xi in x] if self.training else x
        one2one = self.forward_head(x_detach, **self.one2one)
        preds = {"one2many": preds, "one2one": one2one}
    # 학습 중이면 날것 그대로 손실 계산으로 넘김 (사람 탐지 예측에서는 해당 없음)
    if self.training:
        return preds
#@key 예측: 박스 좌표 + 확률 → [1, 84, 6300]
    # 5-4 _inference
    y = self._inference(preds["one2one" if self.end2end else "one2many"] if "one2one" in preds else preds)
#@dim YOLO26용 후처리, 결과 돌려주기
    # NMS 없는 모델이면 여기서 후처리 (YOLO11은 해당 없음 → NMS는 6절)
    if self.end2end:
        y = self.postprocess(y.permute(0, 2, 1))
    # (예측 결과, 날것) 함께 돌려줌
    return y if self.export else (y, preds)
```

### 5-4. `Detect._inference` · `_get_decode_boxes`

```python
#@key 박스 4개 + 확률 80개를 이어 붙임
def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    # 박스 좌표 [1, 4, 6300] (픽셀 단위, 중심x·중심y·너비·높이)
    dbox = self._get_decode_boxes(x)
    # 점수 → sigmoid로 0~1 확률. 박스 4 + 확률 80 = [1, 84, 6300]
    return torch.cat((dbox, x["scores"].sigmoid()), 1)

#@dim 입력 크기가 바뀔 때만 칸 중심 좌표 새로 만들기 (5-5)
def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
    # 가장 큰 지도의 모양 [1, 64, 60, 80]
    shape = x["feats"][0].shape
    # 처음이거나 입력 크기가 바뀌었으면 칸 중심 좌표·stride를 새로 만듦 (영상처럼 같은 크기면 한 번만)
    if self.dynamic or self.shape != shape:
        self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
        self.shape = shape

#@key DFL → 칸 중심 ± 거리 → × stride = 픽셀 박스
    # ① DFL(5-6): 64채널 → 네 변까지 거리(칸 단위)  ② dist2bbox(5-7): 칸 중심 ± 거리 = 박스  ③ × stride = 픽셀
    dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0), x.get("angle")) * self.strides
    return dbox

#@dim 거리 → 박스 (dist2bbox 부르기만 함)
def decode_bboxes(
    self, bboxes: torch.Tensor, anchors: torch.Tensor, angle: torch.Tensor | None = None
) -> torch.Tensor:
    # (중심, 너비·높이) 형식으로 받음 (xywh=True)
    return dist2bbox(bboxes, anchors, xywh=not self.end2end and not self.xyxy, dim=1)
```

### 5-5. `utils/tal.py` `make_anchors` — 칸 중심 좌표 6,300개

```python
#@dim 함수 시작
# feats = 세 지도, strides = [8, 16, 32], grid_cell_offset = 0.5 (칸 한가운데)
def make_anchors(feats, strides, grid_cell_offset=0.5):
    # 칸 중심 좌표 목록, stride 목록
    anchor_points, stride_tensor = [], []
    assert feats is not None
    # 자료형 (16/32비트)
    dtype = feats[0].dtype
#@key 지도마다 칸 한가운데 좌표 (0.5, 1.5, 2.5, …)
    # 지도 3개마다
    for i in range(len(feats)):
        # 8, 16, 32
        stride = strides[i]
        # 60×80, 30×40, 15×20
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        # 칸 한가운데 x 좌표: 0.5, 1.5, 2.5, …
        sx = torch.arange(w).type_as(feats[0]) + grid_cell_offset
        sy = torch.arange(h).type_as(feats[0]) + grid_cell_offset
        # 격자로 펼침
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
        # [칸 수, 2] (x, y) — 지도 칸 단위
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        # 칸마다 stride 값
        stride_tensor.append(feats[0].new_full((h * w, 1), stride, dtype=dtype))
#@plain
    # 이어 붙임 → [6300, 2], [6300, 1]
    return torch.cat(anchor_points), torch.cat(stride_tensor)
```

> YOLO11은 **미리 정해 둔 상자(anchor box)가 없다**. 칸 중심 하나하나가 "여기서 네 변까지 얼마나 먼가"를 직접 예측한다.

### 5-6. `nn/modules/block.py` `DFL` — 16칸 확률 → 거리

```python
#@key 고정 가중치 [0, 1, …, 15] → 기댓값 계산기
class DFL(nn.Module):
    def __init__(self, c1: int = 16):
        super().__init__()
        # 16 → 1채널 1×1 합성곱. 학습하지 않는 고정 층
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        # [0, 1, 2, …, 15]
        x = torch.arange(c1, dtype=torch.float)
        # 합성곱 가중치 = [0…15] → 계산 결과 = Σ(확률 × 칸 번호) = 기댓값
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

#@key 16칸 softmax → 기댓값 = 거리
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b = 1, a = 6300
        b, _, a = x.shape
        # [1, 4변, 16, 6300] → 16칸을 softmax(합 1인 확률) → 기댓값 → [1, 4, 6300] 거리
        # 예: 확률이 3칸 0.6, 4칸 0.4면 거리 = 3.4칸. stride 8 지도면 3.4 × 8 = 27px
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
```

### 5-7. `utils/tal.py` `dist2bbox` — 거리 → 박스

```python
#@key 칸 중심 − (왼, 위) / + (오른, 아래) = 박스
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    # lt = (왼쪽, 위쪽) 거리, rb = (오른쪽, 아래쪽) 거리
    lt, rb = distance.chunk(2, dim)
    # 박스 왼쪽 위 = 칸 중심 − (왼, 위)
    x1y1 = anchor_points - lt
    # 박스 오른쪽 아래 = 칸 중심 + (오른, 아래)
    x2y2 = anchor_points + rb
#@dim (중심, 너비·높이) 형식으로 바꿔 돌려주기
    # 중심·크기 형식을 원하면 (예측 때는 이쪽)
    if xywh:
        # 중심 = 두 꼭짓점의 가운데
        c_xy = (x1y1 + x2y2) / 2
        # 너비·높이 = 두 꼭짓점의 차이
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)
    # 아니면 (왼쪽 위, 오른쪽 아래) 그대로
    return torch.cat((x1y1, x2y2), dim)
```

---

## 6. 후처리: `models/yolo/detect/predict.py` + `utils/nms.py`

### 6-1. `DetectionPredictor.postprocess`

```python
#@dim 함수 시작
# preds = [1, 84, 6300], img = 모델 입력, orig_imgs = 원본 사진
def postprocess(self, preds, img, orig_imgs, **kwargs):
    # 물체별 특징 저장 기능을 켰는지 (기본 꺼짐)
    save_feats = getattr(self, "_feats", None) is not None
#@key NMS: 0.25 미만 버리기 → 사람만 → 겹친 박스 합치기
    preds = nms.non_max_suppression(
        preds,
        # conf = 0.25
        self.args.conf,
        # iou = 예측 기본 0.5 (이만큼 넘게 겹치면 같은 사람으로 보고 지움)
        kwargs.pop("iou", self.args.iou),
        # classes = [0] → 사람만
        self.args.classes,
        self.args.agnostic_nms,
        # 한 장에 최대 300명
        max_det=self.args.max_det,
        nc=0 if self.args.task == "detect" else len(self.model.names),
        end2end=getattr(self.model, "end2end", False),
        rotated=self.args.task == "obb",
        return_idxs=save_feats,
    )

#@dim 원본 사진 형식 맞추기
    # 원본이 텐서로 들어왔으면 넘파이 사진으로 되돌림 (우리는 이미 사진이라 해당 없음)
    if not isinstance(orig_imgs, list):
        orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

#@dim 특징 저장 기능 (기본 꺼짐)
    # 켜져 있으면 NMS가 (결과, 번호)를 돌려주므로 둘로 나눔
    if save_feats:
        obj_feats = self.get_obj_feats(self._feats, preds[1])
        preds = preds[0]

#@plain
    # 박스 좌표를 원본 크기로 되돌리고 Results로 묶음 (7절)
    results = self.construct_results(preds, img, orig_imgs, **kwargs)

#@dim 특징 저장 기능 (기본 꺼짐)
    # 켜져 있으면 결과마다 물체 특징을 붙임
    if save_feats:
        for r, f in zip(results, obj_feats):
            r.feats = f

#@plain
    # 사진마다 Results 하나
    return results
```

### 6-2. `utils/nms.py` `non_max_suppression`

```python
#@dim 함수 머리: 인자와 기본값
def non_max_suppression(
    # 모델 출력 [1, 84, 6300]
    prediction,
    # 점수 기준 (predict에서 0.25를 넘겨줌)
    conf_thres: float = 0.25,
    # 겹침 기준 (예측 기본 0.5)
    iou_thres: float = PREDICT_IOU,
    # [0] = 사람만
    classes=None,
    # True면 종류 구분 없이 NMS
    agnostic: bool = False,
    # True면 한 박스에 여러 종류 허용
    multi_label: bool = False,
    # 한 장 최대 박스 수
    max_det: int = 300,
    nc: int = 0,  # number of classes (optional)
    # 사진당 시간 한도(초)
    max_time_img: float = 0.05,
    # NMS에 넣을 최대 후보 수
    max_nms: int = 30000,
    # 종류별로 박스를 떨어뜨릴 거리(px)
    max_wh: int = 7680,
    # 회전 박스 여부
    rotated: bool = False,
    # NMS 없는 모델(YOLO26) 여부
    end2end: bool = False,
    # 남은 박스의 원래 번호도 돌려줄지
    return_idxs: bool = False,
):
#@dim 값 확인 · 출력 꺼내기 · 사람만 고를 준비
    # 기준값이 0~1 범위인지 확인
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    # 검증 모드 출력은 (추론값, 날것) 쌍이라 추론값만 꺼냄
    if isinstance(prediction, (list, tuple)):  # model in validation mode, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    # classes=[0]을 텐서로
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

#@dim YOLO26처럼 NMS 없는 모델 전용 (YOLO11은 건너뜀)
    # 이미 [박스 수, 6]으로 정리돼 나오는 모델이면 점수·종류로만 거르고 끝
    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output, keepi = [], []
        for pred in prediction:
            mask = pred[:, 4] > conf_thres
            if classes is not None:
                mask &= (pred[:, 5:6] == classes).any(1)
            idx = mask.nonzero(as_tuple=False).view(-1)[:max_det]
            output.append(pred[idx].float())  # half backend outputs: match the float32 rows the NMS branch returns
            keepi.append(idx)
        return (output, keepi) if return_idxs else output

#@dim 묶음 크기, 클래스 수(84 − 4 = 80) 계산
    # 묶음 크기 1
    bs = prediction.shape[0]
    # 클래스 수 = 84 − 4 = 80
    nc = nc or (prediction.shape[1] - 4)
    # 추가 정보 수 (분할 마스크 등, 탐지는 0)
    extra = prediction.shape[1] - nc - 4
    # 추가 정보 시작 위치 = 84
    mi = 4 + nc
#@key 칸마다 80종류 중 최고 점수가 0.25를 넘나
    # → [1, 6300] 참/거짓
    xc = prediction[:, 4:mi].amax(1) > conf_thres
#@dim 번호 기록·시간 제한 등 준비
    # 칸 원래 번호 (특징 저장 기능용)
    xinds = torch.arange(prediction.shape[-1], device=prediction.device).expand(bs, -1)[..., None]
    # 시간 한도 = 2초 + 0.05 × 사진 수
    time_limit = 2.0 + max_time_img * bs
    # 클래스가 여러 개일 때만 한 박스에 여러 종류 허용 (기본 꺼짐)
    multi_label &= nc > 1
#@plain
    # [1, 84, 6300] → [1, 6300, 84]  (한 줄 = 칸 하나)
    prediction = prediction.transpose(-1, -2)

#@dim 결과 담을 곳 준비
    # 시작 시각
    t = time.time()
    # 사진마다 빈 결과 [0, 6] (x1, y1, x2, y2, 점수, 종류)
    output = [torch.zeros((0, 6 + extra), device=prediction.device)] * bs
    # 사진마다 빈 번호 목록
    keepi = [torch.empty(0, dtype=torch.long, device=prediction.device)] * bs
#@key 0.25 넘는 칸만 남기기 (6,300 → 53개)
    # 사진마다
    for xi, (x, xk) in enumerate(zip(prediction, xinds)):
        filt = xc[xi]
        x = x[filt]
        # (중심, 크기) → (왼쪽 위, 오른쪽 아래)
        if not rotated:
            x[:, :4] = xywh2xyxy(x[:, :4])
#@dim 번호 따라 거르기 · 남은 게 없으면 다음 사진
        # 번호도 같이 거름 (특징 저장 기능용)
        if return_idxs:
            xk = xk[filt]

        # 0.25 넘는 칸이 하나도 없으면 다음 사진
        if not x.shape[0]:
            continue

#@plain
        # 박스 4, 클래스 점수 80으로 나눔
        box, cls, mask = x.split((4, nc, extra), 1)

#@dim multi_label: 한 박스에 여러 종류 허용 (기본 꺼짐)
        # 0.25 넘는 (박스, 종류) 쌍을 모두 한 줄씩 만듦
        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
            if return_idxs:
                xk = xk[i]
#@key 기본: 칸마다 가장 점수 높은 종류 하나 → [x1, y1, x2, y2, 점수, 종류]
        # 80종류 중 최고 점수와 그 종류 번호
        else:
            conf, j = cls.max(1, keepdim=True)
            # 그 점수가 0.25를 넘는 것만
            filt = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[filt]
#@dim 번호 따라 거르기
            # 번호도 같이 거름 (특징 저장 기능용)
            if return_idxs:
                xk = xk[filt]

#@key ★ classes=[0] → 종류번호 0(person)만 남김
        if classes is not None:
            filt = (x[:, 5:6] == classes).any(1)
            x = x[filt]
#@dim 번호 거르기 · 비었으면 건너뜀 · 3만 개 넘으면 상위만
            # 번호도 같이 거름
            if return_idxs:
                xk = xk[filt]

        # 남은 박스 수 (사람이 하나도 없으면 다음 사진)
        n = x.shape[0]
        if not n:
            continue
        # 후보가 3만 개 넘으면 점수 높은 3만 개만 (NMS가 너무 느려지지 않게)
        if n > max_nms:
            filt = x[:, 4].argsort(descending=True)[:max_nms]
            x = x[filt]
            if return_idxs:
                xk = xk[filt]

#@key ★ NMS: 겹친 박스 지우기 (53개 → 7명)
        # 종류가 다르면 박스를 7680px씩 떨어뜨려 서로 안 지우게 (사람만 남았으니 모두 0)
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        scores = x[:, 4]
        boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1) if rotated else x[:, :4] + c
        # 점수 높은 박스부터 고르고, 그것과 IoU 0.5 넘게 겹친 박스는 지움
        #   붐비는 군중에서 두 사람 박스가 많이 겹치면 한 명이 지워짐 → 전신이 덜 세는 이유 중 하나
        i = nms_boxes(boxes, scores, iou_thres, rotated=rotated)
        # 최대 300개
        i = i[:max_det]

        output[xi] = x[i]
#@dim 번호 기록 · 시간 넘으면 중단 · 결과 돌려주기
        # 남은 박스의 원래 번호 기록 (특징 저장 기능용)
        if return_idxs:
            keepi[xi] = xk[i].view(-1)
        # 시간 한도를 넘으면 경고하고 중단
        if (time.time() - t) > time_limit:
            LOGGER.warning(f"NMS time limit {time_limit:.3f}s exceeded")
            break

    # 사진마다 [사람 수, 6]
    return (output, keepi) if return_idxs else output
```

---

## 7. 좌표 되돌리기

### 7-1. `DetectionPredictor.construct_result`

```python
#@key 640×480 기준 박스 → 원본 1695×1271 기준
def construct_result(self, pred, img, orig_img, img_path):
    # 7-2 scale_boxes
    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
    # 원본 사진 + 박스(좌표·점수·종류)를 Results로. r.boxes 길이 = 사람 수
    return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6])
```

### 7-2. `utils/ops.py` `scale_boxes`

```python
#@dim 함수 머리
# img1_shape = 모델 입력 크기 (480, 640), boxes = 박스, img0_shape = 원본 크기 (1271, 1695, 3)
def scale_boxes(
    img1_shape: tuple[int, int],
    boxes: torch.Tensor | np.ndarray,
    img0_shape: tuple[int, int],
    ratio_pad: tuple | None = None,
    padding: bool = True,
    xywh: bool = False,
) -> torch.Tensor | np.ndarray:
#@key LetterBox 때 줄인 배율·여백을 다시 계산
    if ratio_pad is None:
        # 배율 = min(480/1271, 640/1695) = 0.378
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        new_h, new_w = (max(round(x * gain), 1) for x in img0_shape[:2])
        gain_y, gain_x = new_h / img0_shape[0], new_w / img0_shape[1]
        # 회색 여백 크기 (1.png는 0)
        pad_x, pad_y = round((img1_shape[1] - new_w) / 2 - 0.1), round((img1_shape[0] - new_h) / 2 - 0.1)
#@dim 배율을 따로 넘겨받았을 때 (우리는 해당 없음)
    # 배율·여백을 직접 넘겨줬으면 그 값을 씀
    else:
        gain_y, gain_x = ratio_pad[0]
        pad_x, pad_y = ratio_pad[1]

#@key 여백 빼고 배율로 나눔 (× 2.65)
    if padding:
        boxes[..., 0] -= pad_x
        boxes[..., 1] -= pad_y
        if not xywh:
            boxes[..., 2] -= pad_x
            boxes[..., 3] -= pad_y
    boxes[..., 0] /= gain_x
    boxes[..., 1] /= gain_y
    boxes[..., 2] /= gain_x
    boxes[..., 3] /= gain_y
#@dim 사진 밖으로 나간 부분 잘라내기
    # 박스가 사진 경계를 넘으면 경계에 맞춰 자름
    return boxes if xywh else clip_boxes(boxes, img0_shape)
```

---

## 8. 정리: 기존 X와 우리 Y의 차이는 어디인가

| | 기존 X (사람) | 우리 Y (머리) | 코드 위치 |
|---|---|---|---|
| 클래스 수 | 80 (그중 0번만 사용) | 1 (head) | yaml `nc`, Detect `self.nc` |
| 칸당 출력 | 4 + 80 = 84 | 4 + 1 = 5 | Detect `self.no` |
| 달라지는 층 | — | cv3 마지막 1×1 합성곱 출력 수 (80 → 1, 학습 때 자동). nc=1로 처음 만들면 cv3 중간 폭이 64지만, `load()`가 COCO의 cv3(폭 80)로 바꿔 끼우고 마지막 1×1만 새로 만들어 결과적으로 이 층 하나만 다름 | 5-1 `nn.Conv2d(c3, self.nc, 1)` |
| 사람만 고르기 | `classes=[0]` (NMS에서) | 필요 없음 | 6-2 |
| 입력 크기 | 640 | 960 (머리가 작아서) | 3-2 |
| 나머지 (전처리·DFL·NMS) | 같음 | 같음 | |

> 2~7절의 Ultralytics 코드는 X와 Y가 **똑같이** 쓴다. Y는 코드를 고친 게 아니라 머리 정답 데이터로 다시 학습한 가중치(`head_base.pt`)를 쓸 뿐이다.

**1.png 결과로 보는 X의 한계**

- 입력 640에서는 7명밖에 못 찾는다. 사진을 0.378배로 줄이면서 뒤쪽 사람이 너무 작아지기 때문이다(입력 1920이면 45명).
- 사람이 겹치면 NMS(6-2)가 한쪽을 지운다. 붐빌수록 전신 박스끼리 많이 겹쳐서 덜 세게 된다.

---

## 부록 A. `predict()` 한 줄이 2~7절을 부르는 과정

> 1절의 `YOLO(...).predict(...)`가 실제로 어떤 순서로 2~7절 함수를 부르는지. 원문 그대로이고 docstring만 뺐다.
> 여기서도 다루지 않는 것: `YOLO("yolo11n.pt")`가 파일을 여는 과정(`AutoBackend`, 수백 줄), 사진 파일을 읽는 데이터 로더(`setup_source`), 결과 저장·화면 표시(`write_results`), 결과 상자 `Results` 클래스.

### A-1. `engine/model.py` `Model.predict`

```python
#@dim 함수 머리 (source = 사진 경로, 나머지 설정은 kwargs로)
def predict(
    self,
    source: str | Path | int | Image.Image | list | tuple | np.ndarray | torch.Tensor | None = None,
    stream: bool = False,
    predictor=None,
    **kwargs: Any,
) -> Iterator[Results | torch.Tensor] | list[Results] | list[torch.Tensor]:
#@dim 사진을 안 넣었으면 예시 사진, 명령줄(yolo predict …)로 불렸는지 확인
    if source is None:
        source = "https://ultralytics.com/images/boats.jpg" if self.task == "obb" else ASSETS
        LOGGER.warning(f"'source' is missing. Using 'source={source}'.")

    # 'yolo predict ...' 명령으로 실행했는지 (우리는 파이썬에서 불러서 False)
    is_cli = (ARGV[0].endswith("yolo") or ARGV[0].endswith("ultralytics")) and any(
        x in ARGV for x in ("predict", "track", "mode=predict", "mode=track")
    )

#@key 예측 설정 합치기 — rect=True가 3-1의 auto=True로 이어짐
    # 예측 기본값: 신뢰도 0.25, 한 번에 1장, rect=True(정사각형으로 안 채움)
    custom = {"conf": 0.25, "batch": 1, "save": is_cli, "mode": "predict", "rect": True, "embed": None}
    kwargs = _handle_deprecation(kwargs)
    prompts = kwargs.pop("prompts", None)  # for SAM-type models
    # 오른쪽일수록 우선: 우리가 넘긴 classes=[0], conf=0.25, imgsz=640이 가장 우선
    args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right
    args["quantize"] = QUANTIZE_ALIASES.get(str(q := args.get("quantize")).lower(), q)  # unset quantize is FP32
#@dim 예측기(DetectionPredictor)를 처음이면 만들고, 있으면 설정만 바꿔 재사용
    current = self.predictor  # this call's predictor, whatever a concurrent call assigns to self.predictor
    if (
        not current
        or (predictor and type(current) is not predictor)
        or any(getattr(current.args, k) != args[k] for k in PREDICTOR_SETUP_KEYS if k in args)
    ):
        # 탐지 모델이면 DetectionPredictor를 만들고 모델을 붙임
        current = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
        current.setup_model(model=self.model, verbose=is_cli)
        self.predictor, cfg = current, current.args
    else:  # the predictor applies cfg under its lock, so concurrent calls never run with each other's args
        base_args = {
            **DEFAULT_CFG_DICT,
            **self.overrides,
            **{k: getattr(current.args, k) for k in PREDICTOR_SETUP_KEYS},
        }
        cfg = get_cfg(base_args, {**custom, **kwargs})
        cfg.show = cfg.show and checks.check_imshow(warn=True)
    if prompts and hasattr(current, "set_prompts"):  # for SAM-type models
        current.set_prompts(prompts)
#@key 예측기 실행 → A-2
    # 우리는 파이썬에서 불렀으니 current(source=사진, ...) 쪽
    return current.predict_cli(source, cfg=cfg) if is_cli else current(source=source, stream=stream, cfg=cfg)
```

### A-2. `engine/predictor.py` `BasePredictor.__call__`

```python
#@key 결과를 목록으로 모아 돌려줌 → A-3
def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
    self.stream = stream
    # stream=True면 하나씩 꺼내 쓰는 방식 (영상용)
    if stream:
        return self.stream_inference(source, model, *args, **kwargs)
    # 우리는 False → 모든 결과를 목록으로. yolo_person.py의 [0]이 이 목록의 첫 번째
    else:
        return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Results into one
```

### A-3. `engine/predictor.py` `BasePredictor.stream_inference` — 실제 반복문

```python
#@dim 준비: 설정 적용, 모델·사진 준비, 저장 폴더, 시간 재기
def stream_inference(self, source=None, model=None, *args, cfg=None, **kwargs):
    # 여러 스레드가 동시에 쓰지 못하게 잠금
    with self._lock:  # for thread-safe inference
        if cfg is not None:  # this call's args and save_dir, set under the lock so concurrent calls keep their own
            keys = ("project", "name", "save_dir", "exist_ok")  # args that set save_dir
            if any(getattr(cfg, k, None) != getattr(self.args, k, None) for k in keys):
                self.save_dir = get_save_dir(cfg)
            self.args = cfg
        if self.args.verbose:
            LOGGER.info("")
        # iou를 안 정했으면 예측 기본값 0.5
        if self.args.iou is None:  # Model.predict resets a reused predictor's args to the defaults
            self.args.iou = self.default_iou
        if self.model is None:
            self.setup_model(model)
        if not getattr(self.model, "base_model", True) and (
            unsupported := [k for k in ("augment", "embed", "visualize") if getattr(self.args, k)]
        ):
            LOGGER.warning(f"{unsupported} not supported by this model (format='{self.model.format}'), ignoring.")
            self.args.augment, self.args.embed, self.args.visualize = False, None, False
        if self.model.format == "pt" and self.model.end2end:
            # Class filtering needs candidates before max_det truncation.
            self.model.model.set_head_attr(max_det=max(self.args.max_det, 300), agnostic_nms=self.args.agnostic_nms)
        # 사진 경로 → 사진을 하나씩 꺼내 주는 데이터 로더 준비
        self.setup_source(source if source is not None else self.args.source)
        if self.args.save or self.args.save_txt:
            (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        self.seen, self.speed, self.pixels, self.windows, self.batch, self._bases = 0, None, None, [], None, set()
        self._sources = {}  # output base of each video path, stream slot, or batch image
        px = 0  # inference pixels summed per image, so a mixed-shape source averages rather than reports its last
        # 전처리·추론·후처리 시간을 각각 재는 타이머 3개
        profilers = (
            ops.Profile(device=self.device),
            ops.Profile(device=self.device),
            ops.Profile(device=self.device),
        )
        dataset = self.dataset
        batches = ((batch, dataset) for batch in dataset)
        if (  # overlap loading with inference; each batch carries a snapshot of the loader's mode, frame and fps
            (self.device.type == "cuda" or getattr(self.model, "prefetch", False))
            and isinstance(dataset, LoadImagesAndVideos)
            and (dataset.nf > dataset.ni or len(dataset) > 1)
        ):
            batches = _prefetch((batch, copy(dataset)) for batch in dataset)
#@dim 반복 시작: 사진(묶음)을 하나씩 꺼냄
        try:
            if self.args.tile and (self.source_type.tensor or self.args.embed or self.args.visualize):
                raise ValueError("tile requires image/video sources without embed or visualize.")
            self.run_callbacks("on_predict_start")
            tile_results, tile_windows = [], []
            # 우리는 사진 1장이라 한 번만 돈다
            for batch, snapshot, tile_context in self._batches(batches):
                self.batch, self.dataset = batch, snapshot
                if not tile_results:
                    start_times = [p.t for p in profilers]
                    pixels = 0
                # paths = 경로, im0s = 원본 사진(BGR), s = 출력 문자열
                paths, im0s, s = self.batch

#@key ① 전처리 (3절) → ② 모델 실행 (4·5절)
                # Preprocess — 3-4 preprocess: LetterBox → BGR→RGB → ÷255 → [1, 3, 480, 640]
                with profilers[0]:
                    im = self.preprocess(im0s)
                # 첫 실행이면 빈 입력으로 한 번 미리 돌려 둠 (속도 안정용)
                if not self.done_warmup:
                    self.model.warmup(im=im)
                    self.done_warmup = True

                # Inference — 모델 실행: 4절 _predict_once → 5절 Detect → [1, 84, 6300]
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embed tensors
                        continue

#@key ③ 후처리 (6·7절): NMS → 원본 좌표 → Results
                # Postprocess
                pixels += im.shape[0] * im.shape[2] * im.shape[3]
                with profilers[2]:
                    # 6-1 postprocess: NMS로 53개 → 7명, 7절로 원본 좌표
                    self.results = self.postprocess(preds, im, im0s)
#@dim 타일링 예측 기능 (기본 꺼짐)
                    if tile_context is not None:
                        from ultralytics.engine.results import Results

                        original, windows, last = tile_context
                        tile_results.extend(self.results)
                        tile_windows.extend(windows)
                        if not last:
                            continue
                        self.batch = original
                        paths, im0s, s = original
                        groups = [([], []) for _ in im0s]
                        for result, (i, window) in zip(tile_results, tile_windows):
                            groups[i][0].append(result)
                            groups[i][1].append(window)
                        self.results = [
                            Results.merge_tiles(
                                r, w, image, self.args.iou, self.args.agnostic_nms, self.args.max_det
                            )
                            for (r, w), image in zip(groups, im0s)
                        ]
                        tile_results.clear()
                        tile_windows.clear()
#@dim 결과 정리 · 속도 기록 · 저장/출력 (우리는 verbose=False라 출력 안 함)
                    self.results = self.finalize_results(self.results)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        px += pixels / n
                        # 결과마다 전처리·추론·후처리에 걸린 시간(ms) 기록
                        self.results[i].speed = {
                            k: (p.t - t) * 1e3 / n
                            for k, p, t in zip(("preprocess", "inference", "postprocess"), profilers, start_times)
                        }
                        self.results[i].colors = getattr(self.model, "colors", None)  # model class colors
                        if (
                            self.args.verbose
                            or self.args.save
                            or self.args.save_txt
                            or self.args.save_crop
                            or self.args.show
                        ):
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
#@key 결과를 하나씩 내보냄 → A-2가 목록으로 모음
                yield from self.results
#@dim 마무리: 영상 파일 닫기, 평균 속도 계산, 저장 위치 출력
        finally:  # also runs when a stream=True consumer abandons the generator or an error aborts the loop
            for v in self.vid_writer.values():
                if isinstance(v, cv2.VideoWriter):
                    v.release()
            batches.close()  # stop the prefetch worker before releasing the capture it reads
            if hasattr(dataset, "close"):  # stop LoadStreams threads and release source captures
                dataset.close()

        # Final results, under the lock: seen is reset by every run, so reading it outside could divide this run's
        # profilers by a concurrent run's count. px and profilers are locals and are already private to this run.
        if seen := self.seen:
            t = tuple(x.t / seen * 1e3 for x in profilers)  # speeds per image
            self.speed = dict(zip(("preprocess", "inference", "postprocess"), t))
            self.pixels = round(px / seen)  # mean area, pairing with speeds that are themselves per-image means
            if self.args.verbose:
                LOGGER.info(
                    f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                    f"{(min(self.args.batch, seen), getattr(self.model, 'channels', 3), *im.shape[2:])}" % t
                )
        if self.args.show:
            cv2.destroyAllWindows()  # close any open windows
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
    self.run_callbacks("on_predict_end")  # outside the lock, so a callback can predict with this model
```

---

## 부록 B. 층 구현: yaml에 나오는 층 종류 (`nn/modules/conv.py`, `block.py`)

> 2절 yaml의 `Conv`, `C3k2`, `SPPF`, `C2PSA`, `Concat`과, Detect 안의 `DWConv`가 실제로 무엇인지. C3k2는 C2f를 물려받고 Bottleneck·C3k를 쓰므로 그것들도 함께 실었다.

### B-1. `autopad` · `Conv` — 모든 층의 기본 벽돌

```python
#@key 패딩 자동 계산: 3×3이면 1칸 → 보폭 1이면 크기 그대로, 보폭 2면 정확히 반
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # 띄엄띄엄 보는 합성곱(dilation)이면 실제 필터 크기로 환산 (YOLO11은 1이라 해당 없음)
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    # 패딩 = 필터 크기 // 2 (3×3 → 1, 1×1 → 0)
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


#@key Conv = 합성곱 → BatchNorm → SiLU 활성화
class Conv(nn.Module):
    # 기본 활성화 함수 SiLU (x × sigmoid(x))
    default_act = nn.SiLU()  # default activation

    # c1 = 입력 채널, c2 = 출력 채널, k = 필터 크기, s = 보폭(2면 크기 반)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        # 합성곱 (bias 없음: 바로 뒤 BatchNorm이 대신함)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        # BatchNorm: 채널마다 값의 평균·분산을 고르게 맞춤
        self.bn = nn.BatchNorm2d(c2)
        # 활성화 함수 (act=False면 그대로 통과)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        # 합성곱 → BatchNorm → SiLU
        return self.act(self.bn(self.conv(x)))

#@dim 합친 모델용: BatchNorm을 합성곱에 미리 합쳐 둔 뒤 쓰는 빠른 경로
    def forward_fuse(self, x):
        return self.act(self.conv(x))
```

### B-2. `DWConv` · `Concat`

```python
#@key DWConv = 채널마다 따로 거는 합성곱 (계산량이 훨씬 적음, Detect의 cv3에서 사용)
class DWConv(Conv):
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        # groups = 입력·출력 채널의 최대공약수 → 입력=출력이면 채널마다 필터 하나씩
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


#@key Concat = 지도 여러 장을 채널 방향으로 이어 붙임 (넥에서 사용)
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        # 1 = 채널 축 → [1, 64, 60, 80] + [1, 128, 60, 80] = [1, 192, 60, 80]
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        return torch.cat(x, self.d)
```

### B-3. `Bottleneck` — C3k2 안에 들어가는 작은 블록

```python
#@key 3×3 합성곱 두 번 + 지름길(입력을 그대로 더함)
class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple = (3, 3), e: float = 0.5):
        super().__init__()
        # 중간 채널 = 출력 × e
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        # 입력·출력 채널이 같고 shortcut=True면 지름길 사용
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 지름길: 입력 x를 결과에 더함 → 깊어져도 학습이 잘 됨 (ResNet 방식)
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
```

### B-4. `C2f` · `C3` · `C3k` · `C3k2` — yaml의 `C3k2`

```python
#@key C2f = 반으로 나눠 한쪽만 Bottleneck을 거치고, 중간 결과를 전부 이어 붙임
class C2f(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
        super().__init__()
        # 중간 채널 = 출력 × 0.5
        self.c = int(c2 * e)  # hidden channels
        # 1×1로 2c 채널을 만들어 반으로 나눌 준비
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        # 이어 붙인 (2+n)c 채널 → 출력 c2 채널
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # Bottleneck n개 (C3k2는 아래에서 이걸 바꿔 끼움)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 반으로 나눔 → [a, b]
        y = list(self.cv1(x).chunk(2, 1))
        # 마지막 것을 Bottleneck에 통과시키며 결과를 계속 붙임 → [a, b, m1(b), m2(m1(b)), …]
        y.extend(m(y[-1]) for m in self.m)
        # 전부 이어 붙여 1×1로 정리
        return self.cv2(torch.cat(y, 1))

#@dim 내보내기용으로 chunk 대신 split을 쓰는 같은 계산
    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


#@dim C3 = 두 갈래(한쪽만 Bottleneck) 후 합침 — C3k의 부모
class C3(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 갈래 1: 1×1 → Bottleneck들, 갈래 2: 1×1만 → 이어 붙여 1×1
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


#@dim C3k = C3인데 Bottleneck 필터 크기를 정할 수 있음 (yaml에서 True라고 쓴 C3k2가 이걸 씀)
class C3k(C3):
    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


#@key C3k2 = C2f에서 안쪽 블록만 바꿔 끼운 것 (YOLO11의 주력 블록)
class C3k2(C2f):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        # yaml의 [256, False, 0.25]에서 False/True가 이 값
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        # 뼈대는 C2f 그대로 (반으로 나누기 → 통과 → 이어 붙이기)
        super().__init__(c1, c2, n, shortcut, g, e)
        # 안쪽 블록만 교체: attn이면 Bottleneck+주의, c3k=True면 C3k, 아니면 Bottleneck
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )
```

### B-5. `SPPF` — yaml 9번 층

```python
#@key 같은 5×5 최대 풀링을 3번 겹쳐 5·9·13 크기 범위를 한꺼번에 봄
class SPPF(nn.Module):
    export = False

    def __init__(self, c1: int, c2: int, k: int = 5, n: int = 3, shortcut: bool = False):
        super().__init__()
        # 채널을 반으로 줄여 계산량 절약
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=False)
        # 원본 + 풀링 3번 = 4개를 이어 붙인 채널 → 출력
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        # 5×5 최대 풀링, 보폭 1 → 크기는 그대로
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        k = self.m.kernel_size
        # Faster on CPU only, CUDA and exports keep square pooling; export first so torch.fx tracing never reads device
        # CPU에서는 5×5 풀링을 가로 1×5, 세로 5×1로 나눠서 함 (결과 같음, 더 빠름)
        separable = not self.export and x.device.type == "cpu"
        # 풀링을 3번 겹침: 5×5 → 9×9 → 13×13 범위를 본 것과 같음
        for _ in range(getattr(self, "n", 3)):
            # Pool width first to preserve square pooling's row-major tie breaking during backpropagation.
            y.append(
                F.max_pool2d(F.max_pool2d(y[-1], (1, k), 1, (0, k // 2)), (k, 1), 1, (k // 2, 0))
                if separable
                else self.m(y[-1])
            )
        # 4개를 이어 붙여 1×1로 정리
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y
```

### B-6. `Attention` · `PSABlock` · `C2PSA` — yaml 10번 층

```python
#@key Attention = 지도의 모든 칸끼리 "얼마나 관련 있나"를 계산해 정보를 섞음
class Attention(nn.Module):
    format = None

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        # 머리(head) 수만큼 나눠서 따로 계산
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        # 점수를 키 크기의 제곱근으로 나눔 (값이 너무 커지지 않게)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        # 1×1 합성곱 하나로 q(질문), k(열쇠), v(값)를 한꺼번에 만듦
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        # 위치 정보(positional encoding)를 3×3 채널별 합성곱으로 더함
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # 칸 수 (640×480 입력이면 10번 층은 20×15 = 300칸)
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
#@dim 애플 기기(CoreML) 내보내기용 경로
        if self.format == "coreml" and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1))
            x = x.transpose(-2, -1).reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
#@key 주의 계산: (q · k) → softmax → v에 가중합
        else:
            # 칸 300개 × 300개의 관련도 표 → softmax로 합이 1인 비율로
            attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
            # 관련도 비율로 v를 섞고, 위치 정보를 더함
            x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


#@key PSABlock = 주의 + 1×1 합성곱 두 개, 둘 다 지름길로 더함 (트랜스포머 블록과 같은 모양)
class PSABlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        # 채널을 2배로 늘렸다가 다시 줄이는 작은 신경망
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


#@key C2PSA = 반으로 나눠 한쪽만 PSABlock을 거친 뒤 합침
class C2PSA(nn.Module):
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 반으로 나눔: a는 그대로, b만 주의 블록 통과
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        # 다시 합쳐 1×1로 정리
        return self.cv2(torch.cat((a, b), 1))
```

---

## 부록 C. 도우미 함수 (`utils/ops.py`, `utils/nms.py`)

```python
#@key xywh2xyxy: (중심, 너비·높이) → (왼쪽 위, 오른쪽 아래) — 6-2에서 사용
def xywh2xyxy(x):
    # 목록이면 배열로
    if isinstance(x, (list, tuple)):
        x = np.asarray(x, dtype=np.float32)  # float so odd integer boxes keep fractional centers
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = empty_like(x)  # faster than clone/copy
    # 중심
    xy = x[..., :2]  # centers
    # 너비·높이의 절반
    wh = x[..., 2:] / 2  # half width-height
    # 왼쪽 위 = 중심 − 절반
    y[..., :2] = xy - wh  # top left xy
    # 오른쪽 아래 = 중심 + 절반
    y[..., 2:] = xy + wh  # bottom right xy
    return y


#@key clip_boxes: 박스가 사진 밖으로 나가면 경계에 맞춰 자름 — 7-2에서 사용
def clip_boxes(boxes, shape):
    # 원본 높이 1271, 너비 1695
    h, w = shape[:2]  # supports both HWC or HW shapes
    if isinstance(boxes, torch.Tensor):  # faster individually
        # x는 0~너비, y는 0~높이 범위로 자름
        if NOT_MACOS14 and not (boxes.device.type == "mps" and check_version(TORCH_VERSION, "<2.5.0")):
            boxes[..., 0].clamp_(0, w)  # x1
            boxes[..., 1].clamp_(0, h)  # y1
            boxes[..., 2].clamp_(0, w)  # x2
            boxes[..., 3].clamp_(0, h)  # y2
#@dim 맥 GPU 버그 회피용 · 넘파이 배열일 때 (우리는 해당 없음)
        else:  # MPS strided in-place bug on macOS 14 or torch<2.5
            boxes[..., 0] = boxes[..., 0].clamp(0, w)
            boxes[..., 1] = boxes[..., 1].clamp(0, h)
            boxes[..., 2] = boxes[..., 2].clamp(0, w)
            boxes[..., 3] = boxes[..., 3].clamp(0, h)
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, w)  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, h)  # y1, y2
#@plain
    return boxes


#@key nms_boxes: 실제 NMS 계산은 torchvision 라이브러리에 맡김 — 6-2에서 사용
def nms_boxes(boxes: torch.Tensor, scores: torch.Tensor, iou: float, rotated: bool = False) -> torch.Tensor:
    # 회전 박스용 (우리는 해당 없음)
    if rotated:
        return TorchNMS.fast_nms(boxes, scores, iou, iou_func=batch_probiou)
    # Keep torchvision lazy and preserve XPU's native torch path.
    # torchvision이 있으면 그걸로: 점수 높은 순으로 고르고 IoU 0.5 넘게 겹친 건 버림
    if boxes.device.type != "xpu" and "torchvision" in sys.modules:
        import torchvision

        return torchvision.ops.nms(boxes, scores, iou)
    # 없으면 Ultralytics가 직접 만든 같은 계산
    return TorchNMS.nms(boxes, scores, iou)
```
