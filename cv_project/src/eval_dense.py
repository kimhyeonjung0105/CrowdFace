"""초밀집 평가: 위험한 밀도에서 머리 탐지가 어디서 무너지는가 — JHU-CROWD++ 시험 1,600장.

usage: python src/eval_dense.py [최대장수]      → results/eval_dense/summary.json, heads.npz(머리별 기록, GitHub 제외)
데이터: data/jhu/jhu_crowd_v2.0/  (JHU-CROWD++ v2.0, 학술·비상업 전용 — 이미지·라벨을 GitHub에 올리지 않는다)
  gt/<이름>.txt 한 줄 = x y w h 가림(1 보임·2 일부·3 완전) 흐림(0·1)   x,y = 머리 위치, w,h = 대략의 머리 크기

왜 필요한가
  지금까지 시험한 영상(HT21-03)은 최대 ㎡당 2.6명. 경보가 필요한 ㎡당 5명 이상(이태원 7.7~10.7명)에서는 한 번도 확인하지 않았다.
  그런 밀도에서는 머리끼리 붙어 탐지가 무너질 수 있다 → 무너진다면 밀도 지도 방식(DM-Count)과 섞어야 한다(다음 단계).
비교하는 방법 (같은 사진, 학습·기준 정하기에 JHU를 쓰지 않음 = 처음 보는 데이터)
  head_full  : 우리 머리 모델, 사진 전체를 960으로 (기준 0.15, CrowdHuman에서 정함)
  head_tile2 : 우리 머리 모델 + 타일링 2×2 (기준 0.25, CrowdHuman에서 정함)
  p2pnet     : 기존 머리 '점' 방법 (ShanghaiTech A로 학습, 초밀집 사진에 강함)   ← 9주차에 돌린 참조 코드
  dmcount    : 기존 밀도 지도 방법 (UCF-QNRF로 학습) — 위치가 없고 인원만 나옴     ← 9주차에 돌린 참조 코드
  body       : 기존 기능 X, 전신 YOLO (참고)
'빽빽함' c = 가장 가까운 머리 3개까지 평균 거리 ÷ 머리 폭  (사진 속에서 머리가 머리 몇 개 간격으로 붙어 있나, 크기와 무관한 값)
  c가 작을수록 빽빽하다. ㎡당 인원으로 바꾸는 근사는 아래 floor_scale()(HT21-03의 바닥 정보로 맞춤).
찾음 판정: 예측(박스 중심 또는 점)이 정답 머리 위치에서 머리 대각선의 절반 안 + 1:1 짝 (NWPU-Crowd 위치 평가와 같은 방식의 넓은 기준)
"""
import sys, json, types, pathlib as _pl
import numpy as np, torch
ROOT = _pl.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from cvio import imread
from tile_detect import detect_tiled
JHU = ROOT / "data" / "jhu" / "jhu_crowd_v2.0" / "test"
OUT = ROOT / "results" / "eval_dense"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAXDET = 10000                       # 한 장에 9,000명이 넘는 사진이 있어 기본 300·1000으로는 모자람
COUNT_BINS = [(0, 50), (51, 200), (201, 500), (501, 1000), (1001, 3000), (3001, 10**6)]
C_BINS = [(0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, np.inf)]
S_BINS = [(0, 8), (8, 16), (16, 32), (32, np.inf)]   # 머리 폭(원본 픽셀)


def knn_mean(p, k=3):
    """점마다 가까운 이웃 k개까지 평균 거리 (GPU 행렬 거리)"""
    if len(p) <= k:
        return np.full(len(p), np.inf)
    t = torch.tensor(p, dtype=torch.float32, device=DEV)
    return torch.cdist(t, t).topk(k + 1, largest=False).values[:, 1:].mean(1).cpu().numpy()


def match(gt_xy, r, pred_xy):
    """정답 머리마다 찾았는지: 거리 ≤ r 인 예측과 가까운 짝부터 1:1"""
    hit = np.zeros(len(gt_xy), bool)
    if len(gt_xy) == 0 or len(pred_xy) == 0:
        return hit
    g = torch.tensor(gt_xy, dtype=torch.float32, device=DEV); p = torch.tensor(pred_xy, dtype=torch.float32, device=DEV)
    D = torch.cdist(g, p); ok = D <= torch.tensor(r, dtype=torch.float32, device=DEV)[:, None]
    gi, pi = torch.nonzero(ok, as_tuple=True); d = D[gi, pi]
    order = torch.argsort(d).cpu().numpy(); gi = gi.cpu().numpy(); pi = pi.cpu().numpy()
    used = np.zeros(len(pred_xy), bool)
    for k in order:
        if not hit[gi[k]] and not used[pi[k]]:
            hit[gi[k]] = used[pi[k]] = True
    return hit


# ---------------- 기존 방법 불러오기 (crowd_density.py · crowd_heads.py와 같은 설정) ----------------
NORM = None
def load_refs():
    global NORM
    import torchvision
    from torchvision import transforms
    NORM = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    import torch.utils.model_zoo as mz; mz.load_url = lambda *a, **k: {}
    sys.path.insert(0, str(ROOT / "refs/DM-Count"))
    from models import vgg19
    dm = vgg19(); dm.load_state_dict(torch.load(ROOT / "refs/DM-Count/pretrained_models/model_qnrf.pth", map_location="cpu")); dm.eval().to(DEV)
    for k in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
        del sys.modules[k]                                       # 두 저장소가 같은 이름 'models'를 써서 비움
    sys.path.remove(str(ROOT / "refs/DM-Count"))
    _vgg = torchvision.models.vgg16_bn; torchvision.models.vgg16_bn = lambda pretrained=False, **k: _vgg(weights=None)
    ver = torchvision.__version__; torchvision.__version__ = "0.99"
    sp = types.ModuleType("scipy"); spo = types.ModuleType("scipy.optimize"); spo.linear_sum_assignment = None; sp.optimize = spo
    saved = {k: sys.modules.get(k) for k in ("scipy", "scipy.optimize")}; sys.modules.update({"scipy": sp, "scipy.optimize": spo})
    sys.path.insert(0, str(ROOT / "refs/CrowdCounting-P2PNet"))
    import models.vgg_ as V
    f = V.vgg16_bn; V.vgg16_bn = lambda pretrained=False, **k: f(pretrained=False, **k)
    from models import build_model
    class A: backbone = "vgg16_bn"; row = 2; line = 2
    p2 = build_model(A()); p2.load_state_dict(torch.load(ROOT / "refs/CrowdCounting-P2PNet/weights/SHTechA.pth", map_location="cpu")["model"]); p2.eval().to(DEV)
    torchvision.__version__ = ver
    for k, v in saved.items():
        if v is None: sys.modules.pop(k, None)
        else: sys.modules[k] = v
    return dm, p2


def resized(img_bgr, max_side, mult):
    """긴 변 ≤ max_side, 가로·세로를 mult의 배수로 (VGG 계열이 큰 사진에서 메모리가 넘치지 않게)"""
    from PIL import Image
    H, W = img_bgr.shape[:2]; s = min(1.0, max_side / max(H, W))
    w2, h2 = max(mult, int(W * s) // mult * mult), max(mult, int(H * s) // mult * mult)
    im = Image.fromarray(img_bgr[:, :, ::-1]).resize((w2, h2), Image.BILINEAR)
    return NORM(im).unsqueeze(0).to(DEV), W / w2, H / h2


@torch.no_grad()
def run_dm(dm, img):
    x, _, _ = resized(img, 2048, 16)
    return float(dm(x)[0].sum())


@torch.no_grad()
def run_p2(p2, img):
    x, sx, sy = resized(img, 2048, 128)
    o = p2(x); sc = torch.softmax(o["pred_logits"], -1)[0, :, 1]
    return (o["pred_points"][0][sc > 0.5].cpu().numpy() * [sx, sy])


def floor_scale():
    """HT21-03(바닥 정보가 있는 영상)의 정답 머리로 '빽빽함 c' ↔ ㎡당 인원 관계를 맞춘다.
    머리가 촘촘한 격자처럼 서 있으면 이웃 간격 d(m)와 밀도는 ρ ≈ (1.075 / d)² (육각 격자).
    사진 속 간격 = c × 머리 폭(px)이고, 그 머리 폭이 바닥에서 몇 m에 해당하는지(w_eff)를 HT21-03에서 구한다.
    → ρ ≈ (1.075 / (c · w_eff))²   (이 카메라 각도 기준의 근사. 비스듬할수록 같은 ρ에서 c가 더 작아진다)"""
    from crowd_alarm import Floor
    seq = ROOT / "data/ht21/HT21/train/HT21-03"
    cfg = json.loads((seq / "img1.floor.json").read_text(encoding="utf-8")); fl = Floor(cfg["floor"], cfg["size"])
    gt = np.loadtxt(seq / "gt/gt.txt", delimiter=","); gt = gt[gt[:, 7] <= 3]
    ratios = []
    for f in range(1, 1001, 10):
        g = gt[gt[:, 0] == f][:, 2:6]
        if len(g) < 10: continue
        cxy = np.c_[g[:, 0] + g[:, 2] / 2, g[:, 1] + g[:, 3] / 2]
        feet = np.c_[cxy[:, 0], g[:, 1] + g[:, 3] + 6 * g[:, 3]]
        fxy = fl.to_floor(feet)
        inside = (fxy[:, 0] > 1) & (fxy[:, 0] < fl.w - 1) & (fxy[:, 1] > 1) & (fxy[:, 1] < fl.d - 1)
        c = knn_mean(cxy) / g[:, 2]; dm = knn_mean(fxy)
        ratios += list((dm / c)[inside & np.isfinite(c)])
    w_eff = float(np.median(ratios))
    return {"w_eff_m": w_eff, "rho_of_c": {f"{c}": round((1.075 / (c * w_eff)) ** 2, 2) for c in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)},
            "c_of_rho": {f"{r}": round(1.075 / (np.sqrt(r) * w_eff), 2) for r in (2, 3, 4, 5, 7.7, 10)}}


def main():
    from ultralytics import YOLO
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    lab = [l.split(",") for l in (JHU / "image_labels.txt").read_text().split("\n") if l.strip()][:limit]
    head = YOLO(str(ROOT / "models/head_base.pt")); body = YOLO(str(ROOT / "models/yolo11n.pt"))
    dm, p2 = load_refs()
    M = ["head_full", "head_tile2", "p2pnet", "dmcount", "body"]
    counts = {k: [] for k in M + ["gt"]}; rec = {"c": [], "w": [], "eff": [], "img": []}; hits = {k: [] for k in ("head_full", "head_tile2", "p2pnet")}
    for i, l in enumerate(lab):
        img = imread(JHU / "images" / f"{l[0]}.jpg"); H, W = img.shape[:2]
        g = np.loadtxt(JHU / "gt" / f"{l[0]}.txt", ndmin=2).reshape(-1, 6)
        xy = g[:, :2]; w = np.maximum(g[:, 2], 1); r = np.maximum(0.5 * np.hypot(g[:, 2], g[:, 3]), 4.0)   # 머리 대각선의 절반, 아주 작은 머리도 최소 4px
        counts["gt"].append(int(l[1]))
        rec["c"].append(knn_mean(xy) / w); rec["w"].append(w); rec["eff"].append(w * 960 / max(H, W)); rec["img"].append(np.full(len(g), i))
        b, _ = detect_tiled(head, img, 1, conf=0.15, max_det=MAXDET); counts["head_full"].append(len(b))
        hits["head_full"].append(match(xy, r, np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]))
        b, _ = detect_tiled(head, img, 2, conf=0.25, max_det=MAXDET); counts["head_tile2"].append(len(b))
        hits["head_tile2"].append(match(xy, r, np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]))
        p = run_p2(p2, img); counts["p2pnet"].append(len(p)); hits["p2pnet"].append(match(xy, r, p))
        counts["dmcount"].append(run_dm(dm, img))
        counts["body"].append(len(body.predict(img, imgsz=960, conf=0.25, classes=[0], max_det=MAXDET, verbose=False)[0].boxes))
        if i % 100 == 0:
            print(f"{i}/{len(lab)}", {k: counts[k][-1] for k in counts}, flush=True)
        torch.cuda.empty_cache()
    gt = np.array(counts["gt"]); res = {"images": len(lab), "methods": {}}
    for k in M:
        c = np.array(counts[k], float)
        res["methods"][k] = {"MAE": float(np.abs(c - gt).mean()), "found_ratio": float(c.sum() / gt.sum()),
                             "by_count": [{"range": f"{lo}-{hi if hi < 10**6 else ''}", "n_img": int(((gt >= lo) & (gt <= hi)).sum()),
                                           "MAE": float(np.abs(c - gt)[(gt >= lo) & (gt <= hi)].mean()),
                                           "ratio": float(c[(gt >= lo) & (gt <= hi)].sum() / max(gt[(gt >= lo) & (gt <= hi)].sum(), 1))} for lo, hi in COUNT_BINS]}
    cc = np.concatenate(rec["c"]); ww = np.concatenate(rec["w"]); H_ = {k: np.concatenate(v) for k, v in hits.items()}
    res["recall_by_crowding"] = {k: [{"c": f"{lo}-{hi}", "heads": int(((cc >= lo) & (cc < hi)).sum()),
                                      "recall": float(H_[k][(cc >= lo) & (cc < hi)].mean())} for lo, hi in C_BINS] for k in H_}
    res["recall_by_crowding_and_size"] = {k: {f"{slo}-{shi}px": [{"c": f"{lo}-{hi}", "heads": int(((cc >= lo) & (cc < hi) & (ww >= slo) & (ww < shi)).sum()),
                                                                  "recall": float(H_[k][(cc >= lo) & (cc < hi) & (ww >= slo) & (ww < shi)].mean()) if ((cc >= lo) & (cc < hi) & (ww >= slo) & (ww < shi)).sum() >= 50 else None}
                                                                 for lo, hi in C_BINS] for slo, shi in S_BINS} for k in H_}
    try:
        res["floor_scale"] = floor_scale()
    except Exception as e:                       # HT21이 없는 PC
        res["floor_scale"] = {"error": str(e)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(OUT / "heads.npz", c=cc, w=ww, eff=np.concatenate(rec["eff"]), img=np.concatenate(rec["img"]),
                        **{f"hit_{k}": v for k, v in H_.items()}, **{f"count_{k}": np.array(v, float) for k, v in counts.items()})
    for k in M:
        m = res["methods"][k]; print(f"{k:11s} MAE {m['MAE']:8.1f}  찾은 비율 {m['found_ratio']:.1%}  " + "  ".join(f"{b['range']} {b['ratio']:.0%}" for b in m["by_count"]))
    for k in H_:
        print(f"{k:11s} 빽빽함별 재현율 " + "  ".join(f"{b['c']} {b['recall']:.0%}" for b in res["recall_by_crowding"][k]))
    print("바닥 환산:", res["floor_scale"])


if __name__ == "__main__":
    main()
