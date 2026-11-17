"""머리 탐지 + 밀도 지도 결합 세기 — 칸마다 '어느 쪽을 믿을지' 고른다.

usage: python src/hybrid_count.py in.jpg out.jpg       (사진 한 장: 칸별로 고른 방법을 색으로 표시)
       다른 코드에서:  from hybrid_count import DensityMap, hybrid_count

왜 필요한가 (eval_dense.py 결과, JHU-CROWD++ 시험 1,600장)
  - 우리 머리 탐지: 200명 이하 사진에서 가장 정확(인원 오차 6·31명) — 그러나 500명 넘으면 절반도 못 셈
  - 밀도 지도(DM-Count): 500명 넘어도 정확 — 그러나 한산한 사진은 30~80% 부풀림
  - 무너지는 원인은 '머리가 작음'과 '머리가 머리 폭보다 좁게 붙음(빽빽함 c<1)'
  → 사진 전체가 아니라 **칸마다** 빽빽한지 보고, 빽빽하면 밀도 지도, 아니면 머리 탐지 수를 쓴다.
칸마다 고르는 규칙 (직접 설계, 기준값은 JHU 검증 500장에서 정함 → eval_hybrid.py)
  1) 칸 안에서 찾은 머리가 2개 이상: 머리 폭 w = 찾은 머리 폭의 중앙값
       빽빽함 c ≈ (칸 넓이 / 인원)^(1/2) / w          (사람 한 명이 차지하는 한 변 길이 ÷ 머리 폭)
       인원은 두 방법 중 큰 값으로 어림(작은 쪽은 놓친 것일 수 있으므로)
       c < C0 이면 밀도 지도, 아니면 머리 탐지
       또 밀도 지도가 탐지보다 R배 넘게 본다면(탐지가 큰 머리 몇 개만 잡고 작은 머리를 놓친 칸) → 밀도 지도
  2) 찾은 머리가 0~1개: 밀도 지도가 M0명 넘게 본다면 '머리가 너무 작아 못 찾은 것' → 밀도 지도, 아니면 머리 탐지
  0) 먼저 사진 전체의 밀도 지도 합이 T_IMG명 이하인 한산한 사진은 머리 탐지만 쓴다
     (밀도 지도는 빈 배경에도 조금씩 사람을 '흩뿌려' 한산한 사진을 부풀리기 때문)
"""
import sys, pathlib as _pl
import numpy as np, torch
ROOT = _pl.Path(__file__).resolve().parent.parent
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# eval_hybrid.py가 JHU 검증 500장에서 고른 값 (results/eval_hybrid/summary.json 'best')
BLOCK, C0, M0, R, T_IMG = 64, 1.5, 4.0, 1e9, 0


class DensityMap:
    """DM-Count(UCF-QNRF 가중치) 밀도 지도. 출력 한 칸 = 입력 8×8 픽셀, 값의 합 = 인원."""
    def __init__(self):
        from torchvision import transforms
        import torch.utils.model_zoo as mz
        mz.load_url = lambda *a, **k: {}                      # ImageNet 초기 가중치는 필요 없음(학습된 가중치로 덮어씀)
        sys.path.insert(0, str(ROOT / "refs/DM-Count"))
        from models import vgg19
        for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[k]
        sys.path.remove(str(ROOT / "refs/DM-Count"))
        self.m = vgg19(); self.m.load_state_dict(torch.load(ROOT / "refs/DM-Count/pretrained_models/model_qnrf.pth", map_location="cpu"))
        self.m.eval().to(DEV)
        self.norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    @torch.no_grad()
    def __call__(self, img_bgr, max_side=2048):
        """→ (밀도 지도 [h, w], 지도 한 칸이 원본에서 몇 픽셀인지 (sx, sy))  — 학습 때와 같은 정규화"""
        from PIL import Image
        H, W = img_bgr.shape[:2]; s = min(1.0, max_side / max(H, W))
        w2, h2 = max(16, int(W * s) // 16 * 16), max(16, int(H * s) // 16 * 16)
        x = self.norm(Image.fromarray(img_bgr[:, :, ::-1]).resize((w2, h2), Image.BILINEAR)).unsqueeze(0).to(DEV)
        d = self.m(x)[0][0, 0].cpu().numpy()
        return d, (W / d.shape[1], H / d.shape[0])


def block_sums(d, k):
    """밀도 지도를 k×k 칸씩 묶어 합 (가장자리 남는 부분도 포함)"""
    h, w = d.shape; H, W = -(-h // k) * k, -(-w // k) * k
    p = np.zeros((H, W), d.dtype); p[:h, :w] = d
    return p.reshape(H // k, k, W // k, k).sum((1, 3))


def hybrid_count(boxes, dmap, scale, block=None, c0=None, m0=None, r=None, t_img=None, return_map=False):
    """boxes [N,4] 원본 xyxy(머리 탐지), dmap·scale = DensityMap 결과 → 결합 인원 (+ 칸별 [탐지 수, 밀도 합, 밀도 지도 썼나])"""
    block = block or BLOCK; c0 = C0 if c0 is None else c0; m0 = M0 if m0 is None else m0
    r = R if r is None else r; t_img = T_IMG if t_img is None else t_img
    k = max(1, block // 8)                                      # 블록 = 입력 block 픽셀 = 밀도 지도 k칸
    dm = block_sums(dmap, k); gh, gw = dm.shape
    cw, ch = k * scale[0], k * scale[1]                         # 원본에서 칸 하나의 가로·세로(px)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2; cy = (boxes[:, 1] + boxes[:, 3]) / 2; bw = boxes[:, 2] - boxes[:, 0]
    gi = np.clip((cy // ch).astype(int), 0, gh - 1); gj = np.clip((cx // cw).astype(int), 0, gw - 1)
    nd = np.zeros((gh, gw)); np.add.at(nd, (gi, gj), 1)
    use_dm = np.zeros((gh, gw), bool)
    area = cw * ch
    sparse = dmap.sum() <= t_img                                # 0) 한산한 사진은 머리 탐지만
    for i, j in zip(*np.nonzero(((nd >= 2) | (dm > m0)) & ~sparse)):
        if nd[i, j] >= 2:
            w = np.median(bw[(gi == i) & (gj == j)])
            c = np.sqrt(area / max(dm[i, j], nd[i, j])) / max(w, 1)
            use_dm[i, j] = (c < c0) or (dm[i, j] > r * nd[i, j] + 1)
        else:
            use_dm[i, j] = True                                 # 밀도 지도는 사람이 있다는데 머리를 못 찾음 = 너무 작은 머리
    n = float(np.where(use_dm, dm, nd).sum())
    return (n, np.stack([nd, dm, use_dm], -1)) if return_map else n


# ---------------- 2판: 원근으로 '그 자리 머리 크기'를 예측해서 고르기 ----------------
# 1판의 한계(eval_hybrid.py): 두 방법이 크게 다를 때 누가 맞는지 못 가림
#   - 한산한 사진에서 밀도 지도가 조명·관중석 무늬를 사람으로 착각 (실제 42명 → 982명)
#   - 빽빽한 칸에서 큰 머리 몇 개만 잡힌 탐지를 믿음 (실제 2,308명 → 1,851명)
# 원리: 같은 사진에서 머리 크기는 화면 아래로 갈수록 거의 일정하게 커진다(원근, crowd_result.py의 slope·horizon과 같은 원리)
#   → 찾은 머리들로 '머리 폭 = a + b·y' 직선을 맞추면, 머리를 못 찾은 칸에서도 원래 머리가 몇 px인지 안다.
#   ① 그 자리 머리가 W_SMALL px보다 작으면(탐지 입력 기준) 탐지가 못 보는 크기 → 밀도 지도
#   ② 밀도 지도가 '빽빽하다'(c < C0)고 하는데 그 자리 머리가 W_BIG px 이상으로 커서 탐지가 충분히 볼 크기이고,
#      탐지는 밀도 지도의 Q배도 못 찾았다 → 밀도 지도의 헛것 → 탐지 (eval_dense: 큰 머리는 빽빽해도 45% 이상 찾음)
#   ③ 그 밖에 밀도 지도가 빽빽하다고 하면 밀도 지도, 아니면 탐지
#   찾은 머리가 5개 미만이면 직선을 못 맞추므로 1판 규칙을 쓴다.
C0_2, W_SMALL, W_BIG, Q = 1.5, 8, 24, 0.1


def head_size_line(boxes, min_n=5):
    """찾은 머리들로 머리 폭 = a + b·y 직선 맞추기 (튀는 값 빼고 한 번 더). 못 맞추면 None"""
    if len(boxes) < min_n:
        return None
    y = (boxes[:, 1] + boxes[:, 3]) / 2; w = boxes[:, 2] - boxes[:, 0]
    a = np.polyfit(y, w, 1)
    for _ in range(2):                                          # 오차가 큰 머리(오탐·특이값)를 빼고 다시
        res = np.abs(w - np.polyval(a, y)); keep = res <= 2.5 * np.median(res) + 1
        if keep.sum() < min_n:
            break
        a = np.polyfit(y[keep], w[keep], 1)
    lo, hi = np.percentile(w, 5) * 0.5, np.percentile(w, 95) * 1.5   # 사진 밖으로 길게 늘리면 엉뚱해지므로 범위 제한
    return lambda yy: np.clip(np.polyval(a, yy), max(lo, 1), hi)


def hybrid_count2(boxes, dmap, scale, c0=None, w_small=None, w_big=None, q=None, block=64, return_map=False):
    c0 = C0_2 if c0 is None else c0; w_small = W_SMALL if w_small is None else w_small
    w_big = W_BIG if w_big is None else w_big; q = Q if q is None else q
    line = head_size_line(boxes)
    if line is None:
        return hybrid_count(boxes, dmap, scale, block, return_map=return_map)
    k = max(1, block // 8); dm = block_sums(dmap, k); gh, gw = dm.shape
    cw, ch = k * scale[0], k * scale[1]; area = cw * ch
    to_input = 960 / max(dmap.shape[0] * scale[1], dmap.shape[1] * scale[0])   # 원본 px → 탐지 모델 입력 px
    cx = (boxes[:, 0] + boxes[:, 2]) / 2; cy = (boxes[:, 1] + boxes[:, 3]) / 2
    gi = np.clip((cy // ch).astype(int), 0, gh - 1); gj = np.clip((cx // cw).astype(int), 0, gw - 1)
    nd = np.zeros((gh, gw)); np.add.at(nd, (gi, gj), 1)
    w = line((np.arange(gh) + 0.5) * ch)[:, None] * np.ones((1, gw))           # 칸마다 '원래 머리 폭'(원본 px)
    c = np.sqrt(area / np.maximum(dm, 1e-6)) / w                                 # 밀도 지도 말대로라면 빽빽함
    w_in = w * to_input
    dense = c < c0
    fake = dense & (w_in >= w_big) & (nd < q * dm)                               # ② 헛것
    use_dm = (w_in < w_small) & (dm > 0.5) | (dense & ~fake)                     # ①, ③
    n = float(np.where(use_dm, dm, nd).sum())
    return (n, np.stack([nd, dm, use_dm], -1)) if return_map else n


# ---------------- 3판: 역할 나누기 — 인원은 밀도 지도, 위치는 머리 탐지 ----------------
# 1판 결과를 시연 영상(HT21-03)에 대 보니: 머리 탐지 77% vs 밀도 지도 106%(인원 오차 60 vs 18.5)인데 1판은 거의 탐지만 씀(밀도 지도 칸 4%).
# 탐지가 놓치는 원인이 '빽빽함'이 아니라 '생김새'(검은 뒤통수 + 검은 외투)라서 빽빽함 규칙에 걸리지 않음.
# → 고르지 않고 섞는다: 구역마다 탐지한 머리 하나하나에 무게 (밀도 지도 인원 ÷ 탐지 인원)를 준다.
#   인원 합은 밀도 지도를 따르고, 사람이 '어디에' 있는지와 머리 크기(→ 바닥 위 발 위치)는 탐지를 따른다.
#   탐지가 MIN_DET개 미만인 구역은 밀도 지도를 그대로 쓴다(머리가 너무 작아 못 찾은 곳).
REGION, MIN_DET = 256, 3


def hybrid_rescale(boxes, dmap, scale, region=None, min_det=None, return_weights=False):
    """→ (인원, 칸별 인원 지도[밀도 지도 격자]) 또는 return_weights면 (인원, 머리별 무게, 밀도 지도만 쓴 격자 지도)"""
    region = region or REGION; min_det = MIN_DET if min_det is None else min_det
    k = max(1, region // 8); h, w = dmap.shape
    cx = (boxes[:, 0] + boxes[:, 2]) / 2; cy = (boxes[:, 1] + boxes[:, 3]) / 2
    ci = np.clip((cy // scale[1]).astype(int), 0, h - 1); cj = np.clip((cx // scale[0]).astype(int), 0, w - 1)
    det = np.zeros((h, w)); np.add.at(det, (ci, cj), 1)
    out = np.zeros((h, w)); wts = np.zeros(len(boxes)); dm_only = np.zeros((h, w))
    for a in range(0, h, k):
        for b in range(0, w, k):
            dsum = dmap[a:a + k, b:b + k].sum(); nsum = det[a:a + k, b:b + k].sum()
            if nsum >= min_det:
                out[a:a + k, b:b + k] = det[a:a + k, b:b + k] * (dsum / nsum)
                sel = (ci >= a) & (ci < a + k) & (cj >= b) & (cj < b + k); wts[sel] = dsum / nsum
            else:
                out[a:a + k, b:b + k] = dmap[a:a + k, b:b + k]; dm_only[a:a + k, b:b + k] = dmap[a:a + k, b:b + k]
    if return_weights:
        return float(out.sum()), wts, dm_only
    return float(out.sum()), out


if __name__ == "__main__":
    import cv2
    from cvio import imread, imwrite
    from ultralytics import YOLO
    img = imread(sys.argv[1])
    b = YOLO(str(ROOT / "models/head_base.pt")).predict(img, imgsz=960, conf=0.15, max_det=10000, verbose=False)[0].boxes.xyxy.cpu().numpy()
    d, sc = DensityMap()(img)
    n, cells = hybrid_count(b, d, sc, return_map=True)
    k = BLOCK // 8; cw, ch = k * sc[0], k * sc[1]
    over = img.copy()
    for i, j in zip(*np.nonzero(cells[..., 2])):                 # 밀도 지도를 쓴 칸: 보라
        cv2.rectangle(over, (int(j * cw), int(i * ch)), (int((j + 1) * cw), int((i + 1) * ch)), (180, 60, 140), -1)
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    for x0, y0, x1, y1 in b.astype(int):
        cv2.rectangle(img, (x0, y0), (x1, y1), (52, 104, 235), 1)
    imwrite(sys.argv[2], img)
    print(f"머리 탐지 {len(b)}명 · 밀도 지도 {d.sum():.0f}명 → 결합 {n:.0f}명 (밀도 지도를 쓴 칸 {int(cells[..., 2].sum())}/{cells[..., 2].size})")
