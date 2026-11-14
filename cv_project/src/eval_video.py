"""영상 평가: 정답 머리 박스가 있는 군중 영상(HT21)에서 전신 vs 머리 모델의 인원·밀도·경보 단계를 정답과 비교.

usage: python src/eval_video.py <시퀀스 폴더> [--step 1] [--imgsz 960]
  예) python src/eval_video.py data/ht21/HT21/train/HT21-03
시퀀스 폴더 구성 (MOTChallenge 형식): img1/000001.jpg …, gt/gt.txt, seqinfo.ini
  gt.txt 한 줄 = 프레임, ID, x, y, w, h, 사용여부, 클래스, 보임정도   (클래스 1 = 사람 머리만 씀)
  img1.floor.json 이 있으면(바닥 4점·실제 크기) ㎡당 밀도와 경보 단계까지 비교, 없으면 인원만.

왜 이 평가가 필요한가
  - CrowdHuman 검증(사진)은 학습과 같은 출처. HT21은 다른 출처의 '위에서 내려다본 영상'이라
    처음 보는 환경에서도 머리 세기가 통하는지 확인하는 시험이다(신뢰도 기준도 CrowdHuman에서 정한 값 그대로).
  - 정답 머리로 계산한 밀도·단계가 '실제 위험 단계'. 모델이 그 단계를 얼마나 맞히는지 = 계획서의 '위험 단계 정답 비교'.
출력 results/eval_video/<시퀀스>.json, <시퀀스>_count.png, <시퀀스>_density.png
"""
import sys, json, argparse, pathlib as _pl
import numpy as np, cv2
ROOT = _pl.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from crowd_alarm import Floor, Alarm, Smoother, SMOOTH, level_of, LEVELS, COUNT_CONF, set_levels
from ultralytics import YOLO

MODELS = {   # 이름: (가중치, 신뢰도 기준, 클래스, 입력 크기 — None이면 --imgsz)
    "body": ("yolo11n.pt", 0.25, [0], None),            # 기존 기능 X: COCO 사람(전신), Ultralytics 기본 기준
    "body_1920": ("yolo11n.pt", 0.25, [0], 1920),       # 공정 비교: 전신은 입력을 키우면 더 찾으므로 원본 크기로도 시험
    "head_base": ("head_base.pt", COUNT_CONF, None, None),
    "head_p2": ("head_p2.pt", COUNT_CONF, None, None),
    # 현장 영상 추가 학습(HT21-01·02·04). 기준 0.20은 HT21-02 뒤 20%(검증)에서 calib_conf.py로 정한 값 — 시험 영상 03은 안 씀
    "head_ht21": ("head_ht21.pt", 0.20, None, None),
}
MODELS = {k: v for k, v in MODELS.items() if (ROOT / "models" / v[0]).exists()}
NAME = {"gt": "정답", "body": "전신 YOLO (기존 X)", "body_1920": "전신 YOLO, 입력 1920", "head_base": "머리 · 원본 구조",
        "head_p2": "머리 · P2 구조 (직접 수정)", "head_ht21": "머리 · 현장 영상 추가 학습"}
COLOR = {"gt": "#3d3d3d", "body": "#2a78d6", "body_1920": "#8fb8ea", "head_base": "#eb6834", "head_p2": "#1baf7a", "head_ht21": "#8a4fd3"}
DEMO = (1.0, 1.33, 1.67)   # 시연 모드 기준 = 서울시 기준(3·4·5)의 1/3. 실제 영상이 주의 단계에 못 미쳐 경보 로직을 시험하려고 씀
FOOT = 6.0   # crowd_alarm.py와 같은 발 위치 규칙


def feet_from(boxes, kind):
    """바닥 위 점: 머리는 머리 박스 아래로 머리 높이×FOOT, 전신은 박스 아래 가운데(발)."""
    if len(boxes) == 0:
        return np.zeros((0, 2))
    x = (boxes[:, 0] + boxes[:, 2]) / 2
    if kind == "body":
        return np.stack([x, boxes[:, 3]], 1)
    return np.stack([x, boxes[:, 3] + FOOT * (boxes[:, 3] - boxes[:, 1])], 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("seq"); ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=960); a = ap.parse_args()
    seq = _pl.Path(a.seq); name = seq.name
    imgs = sorted((seq / "img1").glob("*.jpg"))[::a.step]
    fps = 25.0 / a.step
    gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=",")
    gt = gt[gt[:, 7] <= 3]                             # 1 걷는·2 탈것 위·3 서 있는 사람 = 모두 사람 (4는 무시 영역)
    fj = seq / "img1.floor.json"
    floor = None
    if fj.exists():
        cfg = json.loads(fj.read_text(encoding="utf-8")); floor = Floor(cfg["floor"], cfg["size"])

    rec = {k: {"count": [], "roi": [], "maxd": []} for k in ["gt"] + list(MODELS)}
    smooth = {k: Smoother(SMOOTH * fps) for k in rec}     # crowd_alarm.py와 같은 칸별 1초 평균 (정답에도 똑같이)
    def add(k, boxes, kind):
        rec[k]["count"].append(len(boxes))
        if floor:
            cells, n_in = floor.density(feet_from(boxes, kind))
            rec[k]["roi"].append(n_in); rec[k]["maxd"].append(float(smooth[k]([c[0] for c in cells]).max()))

    for p in imgs:                                     # 정답
        g = gt[gt[:, 0] == int(p.stem)][:, 2:6]
        add("gt", np.c_[g[:, :2], g[:, :2] + g[:, 2:]], "head")
    for k, (w, conf, cls, sz) in MODELS.items():
        m = YOLO(str(ROOT / "models" / w))
        for p in imgs:
            r = m.predict(str(p), imgsz=sz or a.imgsz, conf=conf, classes=cls, max_det=1000, verbose=False)[0]
            add(k, r.boxes.xyxy.cpu().numpy(), "body" if k.startswith("body") else "head")
        print(f"{k}: 완료")

    g = {x: np.array(v) for x, v in rec["gt"].items()}
    res = {"seq": name, "frames": len(imgs), "imgsz": a.imgsz, "gt_mean_count": float(g["count"].mean()),
           "conf": {k: v[1] for k, v in MODELS.items()}, "models": {}}
    def alarms(maxd, roi):
        al = Alarm(); [al.update(i / fps, d, n) for i, (d, n) in enumerate(zip(maxd, roi))]; return al.rows
    if floor:
        res["floor_m"] = [floor.w, floor.d]
        res["gt_max_density"] = float(g["maxd"].max())
        res["levels"] = {}
        for tag, ths in (("서울시", (3, 4, 5)), ("시연", DEMO)):
            set_levels(ths)
            res["levels"][tag] = {"thresholds": list(ths),
                                  "gt_frames": {nm: int(sum(level_of(d)[1] == nm for d in g["maxd"])) for nm in ("정상", "주의", "경계", "심각")},
                                  "gt_alarms": alarms(g["maxd"], g["roi"])}
    for k in MODELS:
        c = {x: np.array(v) for x, v in rec[k].items()}
        r = {"count_MAE": float(np.abs(c["count"] - g["count"]).mean()),
             "count_ratio": float(c["count"].sum() / g["count"].sum())}
        if floor:
            r["roi_MAE"] = float(np.abs(c["roi"] - g["roi"]).mean())
            r["maxd_MAE"] = float(np.abs(c["maxd"] - g["maxd"]).mean())
            r["maxd_bias"] = float((c["maxd"] - g["maxd"]).mean())          # 음수 = 밀도를 낮게 봄(위험을 놓칠 쪽)
            r["maxd_corr"] = float(np.corrcoef(c["maxd"], g["maxd"])[0, 1])   # 흐름 일치도: 크기는 달라도 오르내림이 같으면 1에 가까움
            r["roi_corr"] = float(np.corrcoef(c["roi"], g["roi"])[0, 1])
            for tag, ths in (("서울시", (3, 4, 5)), ("시연", DEMO)):
                set_levels(ths)
                r[f"level_agree_{tag}"] = float(np.mean([level_of(x)[1] == level_of(y)[1] for x, y in zip(c["maxd"], g["maxd"])]))
                r[f"alarms_{tag}"] = alarms(c["maxd"], c["roi"])
        res["models"][k] = r
        print(f"{k:10s} 인원 오차 {r['count_MAE']:6.1f}명/프레임  찾은 비율 {r['count_ratio']:.1%}"
              + (f"  ㎡당 밀도 오차 {r['maxd_MAE']:.2f}  흐름 일치 {r['maxd_corr']:.2f}  단계 일치 서울시 {r['level_agree_서울시']:.1%}·시연 {r['level_agree_시연']:.1%}" if floor else ""))
    set_levels((3, 4, 5))
    res["series"] = {k: {x: [float(v) for v in vals] for x, vals in rec[k].items()} for k in rec}   # 프레임별 값 (그래프·보고서용)

    out = ROOT / "results" / "eval_video"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item()), encoding="utf-8")
    plot(rec, fps, out, name, floor)
    print("저장:", out / f"{name}.json")


def plot(rec, fps, out, name, floor):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    t = np.arange(len(rec["gt"]["count"])) / fps
    keys = ["gt"] + list(MODELS)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for k in keys:
        ax.plot(t, rec[k]["count"], color=COLOR[k], lw=2.4 if k == "gt" else 2, ls="--" if k == "gt" else "-", label=NAME[k])
    ax.set_xlabel("시간 (초)"); ax.set_ylabel("화면 전체 인원"); ax.set_ylim(0, max(rec["gt"]["count"]) * 1.35)   # 위쪽은 범례 자리; ax.grid(axis="y", color="#e5e5e5")
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"{name}_count.png", dpi=200); plt.close(fig)
    if not floor:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    import matplotlib.transforms as mt
    tr = mt.blended_transform_factory(ax.transAxes, ax.transData)   # 글자는 그림 오른쪽 바깥(x=축 비율, y=값)
    for th, nm in zip((3, 4, 5), ("주의", "경계", "심각")):         # 실제 기준선(진하게)과 시연 기준선(연하게)
        ax.axhline(th, color="#8a8a8a", lw=1, ls=":"); ax.text(1.01, th, f"{nm} {th}명 (서울시)", transform=tr, va="center", fontsize=8, color="#555")
    for th, nm in zip(DEMO, ("주의", "경계", "심각")):
        ax.axhline(th, color="#c9c9c9", lw=0.8, ls=":"); ax.text(1.01, th, f"시연 {nm} {th:g}", transform=tr, va="center", fontsize=7, color="#999")
    for k in [x for x in keys if x != "body"]:          # 밀도 그래프는 전신 960 생략(너무 낮아 겹침 방지)
        sm = np.convolve(rec[k]["maxd"], np.ones(13) / 13, mode="same")   # 0.5초 이동 평균 (2m 칸이라 0.25 단위로 튐)
        ax.plot(t, sm, color=COLOR[k], lw=2.4 if k == "gt" else 1.6, ls="--" if k == "gt" else "-", label=NAME[k])
    ax.set_xlabel("시간 (초)"); ax.set_ylabel("가장 붐빈 2m 칸 (㎡당 명)"); ax.set_ylim(0, max(5.5, max(max(rec[k]["maxd"]) for k in keys) + 0.3))
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(out / f"{name}_density.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
