"""머리 탐지 학습 결과 보고서(PDF) 만들기.
먼저 실행할 것:
  python src/eval_heads.py                    → results/eval/summary.json        (신뢰도 0.25)
  EVAL_CONF=0.15 python src/eval_heads.py     → results/eval_conf015/summary.json (인원 세기용 기준, 없으면 4-4절 생략)
  python src/eval_video.py data/ht21/HT21/train/HT21-03 → results/eval_video/      (현장 영상 평가, 없으면 5절 생략)
  python src/eval_tiles.py crowdhuman → python src/eval_tiles.py ht21 → results/eval_tiles/ (타일링, 없으면 4-5절 생략)
  python src/eval_fuse.py crowdhuman → python src/eval_fuse.py ht21 → results/eval_fuse/   (전신+머리 결합, 없으면 4-6절 생략)
usage: python src/make_report.py
결과: results/head_detection_report.pdf (그림은 results/report/)
"""
import os, re, csv, json, datetime, pathlib as _pl
import numpy as np, cv2, torch
from cvio import imread, imwrite   # 한글 경로에서도 되는 사진 읽기·쓰기 (cv2.imread/imwrite 대신)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
                                PageBreak, Preformatted, KeepTogether)

ROOT = _pl.Path(__file__).resolve().parent.parent
MODELS = _pl.Path(os.environ.get("EVAL_MODELS", ROOT / "models"))
RUNS = ROOT / "runs" / "head"
FIG = ROOT / "results" / "report"
PDF = ROOT / "results" / "head_detection_report.pdf"
SUMMARY = json.loads((ROOT / "results" / "eval" / "summary.json").read_text(encoding="utf-8"))
_c15 = ROOT / "results" / "eval_conf015" / "summary.json"
SUM15 = json.loads(_c15.read_text(encoding="utf-8")) if _c15.exists() else None      # 신뢰도 0.15로 다시 센 결과
_vid = ROOT / "results" / "eval_video" / "HT21-03.json"
VIDEO = json.loads(_vid.read_text(encoding="utf-8")) if _vid.exists() else None       # HT21-03 현장 영상 평가
_tl = ROOT / "results" / "eval_tiles"
TILES = json.loads((_tl / "crowdhuman.json").read_text(encoding="utf-8")) if (_tl / "crowdhuman.json").exists() else None   # 타일링 (사진)
TILES_V = json.loads((_tl / "ht21.json").read_text(encoding="utf-8")) if (_tl / "ht21.json").exists() else None             # 타일링 (영상)
_fu = ROOT / "results" / "eval_fuse"
FUSE = json.loads((_fu / "crowdhuman.json").read_text(encoding="utf-8")) if (_fu / "crowdhuman.json").exists() else None   # 전신+머리 결합 (사진)
FUSE_V = json.loads((_fu / "ht21.json").read_text(encoding="utf-8")) if (_fu / "ht21.json").exists() else None             # 전신+머리 결합 (영상)
DEV = 0 if torch.cuda.is_available() else "cpu"
FIG.mkdir(parents=True, exist_ok=True)

# ---------- 색·글꼴 (dataviz 기본 팔레트 1~3번, 고정 순서) ----------
COL = {"body": "#2a78d6", "head_base": "#eb6834", "head_p2": "#1baf7a"}
NAME = {"body": "전신 YOLO (기존 X)", "head_base": "머리 · 원본 구조", "head_p2": "머리 · P2 구조 (직접 수정)"}
SHORT = {"body": "전신", "head_base": "머리 base", "head_p2": "머리 P2"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
pdfmetrics.registerFont(TTFont("Malgun", "C:/Windows/Fonts/malgun.ttf"))
pdfmetrics.registerFont(TTFont("MalgunBd", "C:/Windows/Fonts/malgunbd.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Malgun", normal="Malgun", bold="MalgunBd", italic="Malgun", boldItalic="MalgunBd")

MODEL_KEYS = [k for k in ("body", "head_base", "head_p2") if k in SUMMARY]
HEAD_KEYS = [k for k in MODEL_KEYS if k != "body"]


# ---------- 학습 기록 읽기 ----------
def read_run(k):
    p = RUNS / k / "results.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    f = lambda r, c: float(r[c])
    ep = [int(float(r["epoch"])) for r in rows]
    m50 = [f(r, "metrics/mAP50(B)") for r in rows]; m95 = [f(r, "metrics/mAP50-95(B)") for r in rows]
    fit = [0.1 * a + 0.9 * b for a, b in zip(m50, m95)]           # Ultralytics가 best.pt를 고르는 기준
    b = int(np.argmax(fit)); r = rows[b]
    # 학습 시간: resume으로 이어서 학습하면 time 열이 0부터 다시 쌓이므로, 줄어드는 지점마다 앞 구간을 더함
    t = [f(x, "time") for x in rows]
    secs = sum(t[i] for i in range(len(t)) if i == len(t) - 1 or t[i + 1] < t[i])
    log = (ROOT / "runs" / f"train_{k.split('_')[1]}.log")
    txt = log.read_text(encoding="utf-8", errors="ignore") if log.exists() else ""
    opt = re.search(r"optimizer:.*?(\w+)\(lr=([\d.]+)", txt)
    tr = re.search(r"Transferred (\d+)/(\d+)", txt)
    return dict(epochs=ep, m50=m50, m95=m95, best_epoch=ep[b], P=f(r, "metrics/precision(B)"),
                R=f(r, "metrics/recall(B)"), mAP50=m50[b], mAP95=m95[b], hours=secs / 3600,
                optimizer=f"{opt.group(1)} (lr {opt.group(2)})" if opt else "auto",
                transferred=f"{tr.group(1)}/{tr.group(2)}" if tr else "-")


RUN = {k: read_run(k) for k in HEAD_KEYS}


def model_info(path):
    from ultralytics.utils.torch_utils import get_flops
    m = YOLO(str(path)).model
    return sum(p.numel() for p in m.parameters()), get_flops(m, 640)


INFO = {k: model_info(MODELS / ("yolo11n.pt" if k == "body" else f"{k}.pt")) for k in MODEL_KEYS}


# ---------- 그림 1: 학습 곡선 (지표마다 따로, 축 하나씩) ----------
def fig_curves():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), dpi=200)
    for ax, key, title in [(axes[0], "m50", "mAP50"), (axes[1], "m95", "mAP50-95")]:
        for k in HEAD_KEYS:
            if not RUN[k]:
                continue
            x, y = RUN[k]["epochs"], RUN[k][key]
            ax.plot(x, y, color=COL[k], lw=2, label=NAME[k])
            ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xlabel("epoch"); ax.grid(axis="y", color=GRID, lw=0.6); ax.margins(x=0.12)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); p = FIG / "curves.png"; fig.savefig(p); plt.close(fig); return p


# ---------- 그림 2: 붐빈 정도별 '찾은 비율' ----------
def fig_crowd():
    bins = [b["range"] for b in SUMMARY["body"]["by_crowd"]]
    lab = [f"{r.replace('-', '~') if not r.endswith('-') else r[:-1] + '명 이상'}" for r in bins]
    lab = [l if "이상" in l else l + "명" for l in lab]
    n = [b["n_img"] for b in SUMMARY["body"]["by_crowd"]]
    x = np.arange(len(bins)); w = 0.8 / len(MODEL_KEYS)
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=200)
    for i, k in enumerate(MODEL_KEYS):
        v = [100 * b["ratio"] for b in SUMMARY[k]["by_crowd"]]
        bars = ax.bar(x + (i - (len(MODEL_KEYS) - 1) / 2) * w, v, w * 0.92, color=COL[k], label=NAME[k])
        ax.bar_label(bars, labels=[f"{t:.0f}" for t in v], fontsize=7, color=INK2, padding=1)
    ax.axhline(100, color=MUTED, lw=1, ls="--", label="정답 = 100%")
    ax.set_xticks(x, [f"{l}\n(사진 {c}장)" for l, c in zip(lab, n)])
    ax.set_ylabel("찾은 인원 / 실제 인원 (%)"); ax.set_xlabel("사진 한 장의 실제 인원")
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(); p = FIG / "crowd.png"; fig.savefig(p); plt.close(fig); return p


# ---------- 그림 3: 머리 크기별 재현율 ----------
def fig_size():
    rng = [b["range"] for b in SUMMARY[HEAD_KEYS[0]]["recall_by_size"]]
    lab = [r.replace("0-16px", "16px 미만").replace("64-px", "64px 이상").replace("-", "~") for r in rng]
    tot = [b["total"] for b in SUMMARY[HEAD_KEYS[0]]["recall_by_size"]]
    x = np.arange(len(rng)); w = 0.8 / len(HEAD_KEYS)
    fig, ax = plt.subplots(figsize=(7.2, 2.7), dpi=200)
    for i, k in enumerate(HEAD_KEYS):
        v = [100 * (b["recall"] or 0) for b in SUMMARY[k]["recall_by_size"]]
        bars = ax.bar(x + (i - (len(HEAD_KEYS) - 1) / 2) * w, v, w * 0.92, color=COL[k], label=NAME[k])
        ax.bar_label(bars, labels=[f"{t:.0f}" for t in v], fontsize=7, color=INK2, padding=1)
    ax.set_xticks(x, [f"{l}\n(머리 {t:,}개)" for l, t in zip(lab, tot)])
    ax.set_ylabel("찾은 머리 비율 (%)"); ax.set_xlabel("머리 박스 높이 (픽셀)"); ax.set_ylim(0, 105)
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(); p = FIG / "size.png"; fig.savefig(p); plt.close(fig); return p


# ---------- 그림 4: 예시 사진 (같은 사진에 모델별 결과) ----------
def fig_examples(src):
    out = []
    for k in MODEL_KEYS:
        w = MODELS / ("yolo11n.pt" if k == "body" else f"{k}.pt")
        r = YOLO(str(w)).predict(str(src), imgsz=960, conf=0.25, classes=[0] if k == "body" else None,
                                 device=DEV, verbose=False)[0]
        img = r.orig_img.copy(); c = tuple(int(COL[k][i:i + 2], 16) for i in (5, 3, 1))   # BGR
        for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy().astype(int):
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 3)
        s = 900 / img.shape[1]; img = cv2.resize(img, None, fx=s, fy=s)
        p = FIG / f"ex_{k}.jpg"; imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out.append((k, p, len(r.boxes)))
    return out


# ---------- 구조 비교용 텍스트 ----------
def yaml_head(path, strip_comments):
    lines = path.read_text(encoding="utf-8").splitlines()
    i = next(n for n, l in enumerate(lines) if l.startswith("head:"))
    keep = []
    for l in lines[i + 1:]:
        if not l.strip():
            continue
        if strip_comments:
            l = l.split("#")[0].rstrip()
        else:
            l = re.sub(r"#\s*(\d+)\s*", r"# \1 ", l)
        keep.append(l.replace("  - ", "- ").replace('"nearest"', '"n"'))
    return "\n".join(keep)


# ================= PDF =================
S = lambda **kw: ParagraphStyle("x", **{"fontName": "Malgun", "fontSize": 9.5, "leading": 14.5, "textColor": INK, **kw})
ST = {"title": S(fontName="MalgunBd", fontSize=20, leading=26, spaceAfter=4),
      "sub": S(fontSize=10, textColor=INK2, spaceAfter=14),
      "h1": S(fontName="MalgunBd", fontSize=14, leading=20, spaceBefore=10, spaceAfter=6),
      "h2": S(fontName="MalgunBd", fontSize=11, leading=16, spaceBefore=8, spaceAfter=4),
      "body": S(), "bul": S(leftIndent=12, bulletIndent=2),
      "cap": S(fontSize=8, leading=11, textColor=INK2, spaceAfter=8),
      "cell": S(fontSize=8.5, leading=11.5), "cellb": S(fontName="MalgunBd", fontSize=8.5, leading=11.5),
      "code": S(fontSize=6.6, leading=8.6)}
P = lambda t, s="body": Paragraph(t, ST[s])
B = lambda t: Paragraph(t, ST["bul"], bulletText="•")


def table(rows, widths, header=True, zebra=True):
    data = [[c if not isinstance(c, str) else P(c, "cellb" if (header and i == 0) else "cell") for c in r]
            for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor("#c3c2b7")),
          ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.HexColor("#c3c2b7")),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f0ec")))
    if zebra:
        for i in range(2, len(rows), 2):
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f9f9f7")))
    t.setStyle(TableStyle(st)); return t


def pct(v): return f"{100 * v:.1f}%"


def build():
    W = A4[0] - 36 * mm
    story = []
    today = datetime.date.today().isoformat()
    sm = SUMMARY

    # ----- 표지 + 요약 -----
    story += [P("CrowdFace 머리 탐지 학습 결과 보고서", "title"),
              P(f"군중 밀집도 기반 인파 사고 조기 경보 · 추가 기능 Y① &nbsp;|&nbsp; 202104183 김현중, 202204201 김태섭 &nbsp;|&nbsp; {today}", "sub"),
              P("1. 한눈에 보기", "h1"),
              P("기존 YOLO는 사람 <b>전신</b>을 찾는다. 우리는 같은 YOLO 학습 코드를 이해하고 "
                "<b>데이터·출력 클래스·모델 구조</b>를 바꿔 <b>머리</b>를 찾도록 직접 학습시켰다. "
                f"CrowdHuman 검증 사진 {sm['images']:,}장(실제 인원 {sm['gt_total']:,}명)에서 인원을 세어 비교했고"
                + (", 학습에 쓰지 않은 <b>현장 영상(HT21-03, 시부야 교차로)</b>에서도 확인했다." if VIDEO else ".")),
              Spacer(1, 6)]
    hdr = ["모델", "머리 mAP50", "머리 mAP50-95", "인원 오차<br/>(MAE, 명/장)", "찾은 비율<br/>(전체)",
           "찾은 비율<br/>(61명 이상 사진)", "속도<br/>(ms/장)"]
    rows = [hdr]
    for k in MODEL_KEYS:
        r = RUN.get(k)
        rows.append([NAME[k], f"{r['mAP50']:.3f}" if r else "-", f"{r['mAP95']:.3f}" if r else "-",
                     f"{sm[k]['MAE']:.2f}", pct(sm[k]["found_ratio"]), pct(sm[k]["by_crowd"][-1]["ratio"]),
                     f"{sm['speed_ms'][k]:.1f}"])
    story.append(table(rows, [W * .25, W * .11, W * .12, W * .13, W * .11, W * .15, W * .13]))
    story.append(P("mAP: 머리 박스 위치까지 맞혔는지(검증 4,370장, IoU 기준). 인원 오차: 사진마다 |센 인원 - 실제 인원|의 평균. "
                   "찾은 비율: 센 인원 합 ÷ 실제 인원 합(100%에 가까울수록 좋음). "
                   f"속도: RTX 4070, 입력 960, 추론만({sm.get('speed_note', '')}). "
                   "이 표는 Ultralytics 기본 신뢰도 기준 0.25로 센 값(인원 세기용 기준 0.15 결과는 4-4절).", "cap"))
    story += [P("핵심 결과", "h2")] + [B(t) for t in summary_points()]
    story += [Spacer(1, 6), Image(str(fig_crowd()), W, W * 2.9 / 7.2),
              P("검증 사진을 실제 인원으로 묶어, 각 모델이 센 인원 ÷ 실제 인원을 비교했다. 100%보다 낮으면 덜 센 것.", "cap")]

    # ----- 2. 무엇을 바꿨나 -----
    story += [PageBreak(), P("2. 이전 코드와 무엇이 달라졌나", "h1"),
              table([["", "이전 (기존 기능 X)", "현재 (추가 기능 Y①)"],
                      ["찾는 대상", "사람 전신", "사람 머리 (모자·뒤통수 포함)"],
                      ["학습 데이터", "COCO (Ultralytics가 학습해 배포)", "CrowdHuman 15,000장 · 머리 287,106개 (직접 변환)"],
                      ["출력 클래스", "COCO 80종 중 person 사용", "head 1종 (Detect 층 출력 80 → 1)"],
                      ["모델 구조", "YOLO11n 원본 (P3·P4·P5에서 탐지)",
                       "① base: 원본 그대로<br/>② P2: 작은 머리용 P2 탐지층 추가 (직접 수정)"],
                      ["시작 가중치", "COCO 학습 결과", "COCO 가중치에서 전이학습 (맞는 층만 이어받음)"],
                      ["코드", "src/yolo_person.py",
                       "src/convert_crowdhuman.py · models/head_p2.yaml · src/train_head.py · src/eval_heads.py"],
                      ], [W * .15, W * .33, W * .52]),
              Spacer(1, 4),
              P("Ultralytics 라이브러리 코드는 고치지 않았다. 바꾼 것은 <b>무엇을 가르치는지(데이터·클래스)</b>와 "
                "<b>모델 구조 설정 파일</b>이다.", "cap"),
              P("2-1. 데이터 변환: CrowdHuman → YOLO 머리 라벨", "h2"),
              P("CrowdHuman 라벨(.odgt)은 사진 한 장마다 사람 목록이 있고, 사람마다 전신·보이는 부분·머리 박스가 있다. "
                "우리는 머리 박스(hbox)만 골라 YOLO 형식 <font name='Malgun'>“0 cx cy w h”</font>(0~1 비율)로 바꿨다."),
              B("<b>mask 영역</b>(포스터·마네킹, 또는 너무 빽빽해 라벨을 못 단 군중)과 <b>ignore 표시된 머리</b>는 학습에서 뺐다."),
              B("그냥 빼면 그 안의 진짜 머리가 ‘머리 아님’으로 학습된다. Ultralytics에는 무시 영역 기능이 없으므로 "
                "<b>그 영역을 회색(114)으로 칠해 사진에서 지웠다</b>. (학습 105,490곳, 검증 30,000곳)"),
              B("사진 밖으로 나간 박스는 경계에 맞춰 자르고, 2픽셀보다 작으면 뺐다."),
              B("통계: 머리의 약 30%가 높이 16픽셀 미만 → 작은 머리를 잘 찾는 구조가 필요하다는 근거."),
              P("2-2. 모델 구조 변경: P2 탐지층 추가 (models/head_p2.yaml)", "h2"),
              P("원본 YOLO11n은 1/8·1/16·1/32 해상도(P3·P4·P5) 세 곳에서 탐지한다. 16픽셀 머리는 P3 특징 지도에서 2칸뿐이라 놓치기 쉽다. "
                "그래서 백본 2번 층의 <b>1/4 해상도(P2) 특징을 끌어와</b> 한 단계 더 촘촘한 곳에서도 탐지하게 했다. "
                "Ultralytics에는 YOLO11용 P2 설정이 없어(YOLOv8용만 있음) 원본 설정을 직접 고쳤다."),
              ]
    orig = yaml_head(ROOT / ".venv/Lib/site-packages/ultralytics/cfg/models/11/yolo11.yaml", True)
    ours = yaml_head(ROOT / "models/head_p2.yaml", False)
    story.append(table([["원본 yolo11.yaml (head 부분)", "직접 수정한 head_p2.yaml (head 부분, ‘+ 추가’ 표시)"],
                        [Preformatted(orig, ST["code"]), Preformatted(ours, ST["code"])]],
                       [W * .40, W * .60], zebra=False))
    story[-1].setStyle(TableStyle([("VALIGN", (0, 1), (-1, -1), "TOP")]))
    pi = {k: INFO[k] for k in HEAD_KEYS}
    story.append(Spacer(1, 4))
    story.append(table([["", "원본 구조 (base)", "P2 구조"],
                        ["탐지 위치(stride)", "8 · 16 · 32", "4 · 8 · 16 · 32"],
                        ["파라미터 수", f"{pi['head_base'][0]:,}" if 'head_base' in pi else "-",
                         f"{pi['head_p2'][0]:,}" if 'head_p2' in pi else "-"],
                        ["연산량 (GFLOPs, 640 기준)", f"{pi['head_base'][1]:.1f}" if 'head_base' in pi else "-",
                         f"{pi['head_p2'][1]:.1f}" if 'head_p2' in pi else "-"],
                        ["COCO에서 이어받은 가중치", RUN['head_base']['transferred'] if RUN.get('head_base') else "-",
                         RUN['head_p2']['transferred'] if RUN.get('head_p2') else "-"]],
                       [W * .34, W * .33, W * .33]))
    story.append(P("P2 구조는 파라미터는 거의 같지만, 큰 해상도에서 계산하므로 연산량이 늘어난다.", "cap"))

    # ----- 3. 어떻게 학습했나 -----
    story += [PageBreak(), P("3. 어떻게 학습했나", "h1")]
    rb = RUN.get("head_base") or {}; rp = RUN.get("head_p2") or {}
    story.append(table([["항목", "값", "이유"],
                        ["데이터", "학습 15,000장 / 검증 4,370장 (CrowdHuman 공식 분할)", "검증 사진은 학습에 쓰지 않음"],
                        ["입력 크기", "960", "머리가 작아 기본값 640보다 크게"],
                        ["epoch / batch", "30 / base 16 · P2 8", "P2는 batch 16에서 GPU 메모리(12GB)가 모자라 8로 줄임. Ultralytics는 기울기를 모아 유효 batch 64로 맞추므로(16×4 = 8×8) 학습률·최적화 조건은 같음"],
                        ["최적화", rb.get("optimizer", "auto"), "Ultralytics 자동 선택"],
                        ["장비", "RTX 4070 (12GB), Python 3.13, torch 2.15 + CUDA", "Colab 대신 이 PC GPU"],
                        ["학습 시간", f"base {rb.get('hours', 0):.1f}시간 · P2 {rp.get('hours', 0):.1f}시간", "P2는 22 epoch에서 한 번 멈췄다가 resume으로 이어서 학습 (밤에는 다른 작업과 GPU를 나눠 써서 더 느렸음)"],
                        ["가장 좋은 epoch", f"base {rb.get('best_epoch', '-')} · P2 {rp.get('best_epoch', '-')}",
                         "검증 점수가 가장 높은 가중치를 best.pt로 저장"]],
                       [W * .18, W * .47, W * .35]))
    story += [Spacer(1, 6), P("학습 명령 (같은 Ultralytics 학습 코드 model.train)", "h2"),
              Preformatted("# base: COCO 가중치를 불러오면 Detect 층 출력만 1클래스로 새로 만들어짐\n"
                           "YOLO('yolo11n.pt').train(data='head.yaml', epochs=30, imgsz=960, batch=16)\n\n"
                           "# P2: 직접 수정한 구조를 만든 뒤, 모양이 맞는 층만 COCO 가중치를 이식\n"
                           "YOLO('head_p2.yaml').load('yolo11n.pt').train(data='head.yaml', epochs=30, imgsz=960, batch=8)",
                           ST["code"]),
              Spacer(1, 6), P("학습 곡선 (검증 사진 점수, epoch마다)", "h2"),
              Image(str(fig_curves()), W, W * 2.6 / 7.2),
              P("선이 오르다 평평해지면 더 배울 것이 적다는 뜻. 끝의 숫자는 마지막 epoch 점수.", "cap")]

    # ----- 4. 결과 -----
    story += [PageBreak(), P("4. 결과", "h1"), P("4-1. 사람이 많을수록 전신 YOLO는 덜 센다 (1쪽 그림)", "h2")]
    story += [B(t) for t in crowd_points()]
    if len(HEAD_KEYS) >= 1 and "recall_by_size" in sm[HEAD_KEYS[0]]:
        story += [P("4-2. 머리 크기별로 찾은 비율 (P2 구조의 효과)", "h2"),
                  Image(str(fig_size()), W, W * 2.7 / 7.2),
                  P("정답 머리와 IoU 0.5 이상 겹치는 예측이 있으면 ‘찾음’. 작은 머리에서 두 구조의 차이를 본다.", "cap")]
        story += [B(t) for t in size_points()]

    ex = fig_examples(ROOT / "refs/DM-Count/example_images/1.png")
    story += [PageBreak(), P("4-3. 같은 사진, 모델별 결과", "h2"),
              P("DM-Count 예시 사진(학습에 쓰지 않은 사진). 박스 색은 모델 색과 같다.", "cap")]
    iw = W * .49; cells = [[P(f"<b>{NAME[k]}</b> — {n}명", "cell"), Image(str(p), iw, iw * 1271 / 1695)] for k, p, n in ex]
    cells += [[P("", "cell"), Spacer(1, 1)]] * (len(cells) % 2)
    grid = [[cells[i], cells[i + 1]] for i in range(0, len(cells), 2)]
    g = Table(grid, colWidths=[W * .5, W * .5]); g.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.append(g)

    # ----- 4-4. 신뢰도 기준 -----
    if SUM15:
        story += [PageBreak(), P("4-4. 인원 세기용 신뢰도 기준 맞추기 (0.25 → 0.15)", "h2"),
                  P("모델은 박스마다 ‘머리일 확신’(0~1)을 내고, 기준 이상인 박스만 센다. Ultralytics 기본값 0.25는 "
                    "mAP 계산용이라 인원을 덜 세는 쪽(87%)이었다. 그래서 <b>검증 사진에서 센 인원 합 = 실제 인원 합</b>이 되는 "
                    "기준을 찾았더니(src/calib_conf.py) 0.15였다."),
                  Spacer(1, 4)]
        rows = [["모델", "기준", "인원 오차 (MAE)", "찾은 비율 (전체)", "찾은 비율 (61명 이상)"]]
        for k in MODEL_KEYS:
            for c, d in ((0.25, SUMMARY), (0.15, SUM15)):
                rows.append([NAME[k] if c == 0.25 else "", f"{c:.2f}", f"{d[k]['MAE']:.2f}", pct(d[k]["found_ratio"]),
                             pct(d[k]["by_crowd"][-1]["ratio"])])
        story.append(table(rows, [W * .30, W * .10, W * .18, W * .20, W * .22], zebra=False))
        story.append(P("같은 검증 4,370장. 전신 YOLO도 같은 0.15로 다시 셌다(공정 비교). "
                       "기준을 같은 검증 사진으로 정했으므로 이 표의 0.15 결과는 약간 유리하게 나온 값이다(현장 영상 5절은 따로 정한 기준).", "cap"))
        story += [B(t) for t in calib_points()]

    # ----- 4-5. 타일링 -----
    if TILES:
        story += tiles_section(W)

    # ----- 4-6. 전신+머리 결합 -----
    if FUSE:
        story += fuse_section(W)

    # ----- 5. 현장 영상 평가 -----
    if VIDEO:
        story += video_section(W)

    # ----- 6. 정리 -----
    story += [PageBreak(), P(f"{6 if VIDEO else 5}. 정리와 발표 포인트", "h1")] + [B(t) for t in conclusion_points()]
    story += [P("한계", "h2"),
              B("현장 영상 평가는 HT21-03 한 편(40초)이다. 바닥 넓이(10m×28m)는 횡단보도 줄 간격(0.9m 가정)과 맨홀 모양으로 "
                "추정한 값이라 ㎡당 밀도에는 추정 오차가 들어 있다(인원 비교는 영향 없음)." if VIDEO else
                "평가는 CrowdHuman 검증 사진 기준이다. CCTV처럼 위에서 내려다본 사진에서는 따로 확인해야 한다."),
              B("신뢰도 기준은 데이터마다 다르다(CrowdHuman 0.15, 현장 추가 학습 모델 0.20). 새 현장에서는 다시 맞춰야 한다."),
              B("P2는 메모리 때문에 batch 8로 학습했다(base는 16). 유효 batch는 64로 같지만 BatchNorm 통계는 작은 batch로 계산된다."),
              B("정답 인원은 무시 영역을 뺀 머리 라벨 개수다. 회색으로 지운 영역의 사람은 정답에도, 예측에도 들어가지 않는다."),
              P("다음 할 일", "h2"),
              B("어두운 뒤통수 찾기: 현장 영상 추가 학습 데이터를 늘리기 (크기·전신 보완으로는 안 풀림을 확인했으므로)"),
              B("직접 촬영한 영상 + 줄자로 잰 바닥 네 점으로 시연 (바닥 크기 추정 오차 없애기)"),
              B("발표 자료와 시연 영상 정리 (src/crowd_alarm.py, 시연용 기준 표시 유지)")]

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="CrowdFace 머리 탐지 학습 결과 보고서")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("PDF:", PDF)


def footer(c, d):
    c.saveState(); c.setFont("Malgun", 7.5); c.setFillColor(colors.HexColor(MUTED))
    c.drawString(18 * mm, 9 * mm, "컴퓨터비전 미니프로젝트 · CrowdFace 머리 탐지 학습 결과")
    c.drawRightString(A4[0] - 18 * mm, 9 * mm, str(d.page)); c.restoreState()


# ---------- 숫자에서 문장 만들기 ----------
def summary_points():
    sm = SUMMARY; b = sm["body"]; out = []
    best = min(HEAD_KEYS, key=lambda k: sm[k]["MAE"]) if HEAD_KEYS else None
    if best:
        h = sm[best]
        out.append(f"인원 오차(MAE): 전신 {b['MAE']:.1f}명 → 머리 {h['MAE']:.1f}명 "
                   f"(<b>{100 * (1 - h['MAE'] / b['MAE']):.0f}% 감소</b>, {NAME[best]}).")
        out.append(f"61명 이상 붐비는 사진: 전신은 실제 인원의 {pct(b['by_crowd'][-1]['ratio'])}만 셌지만, "
                   f"머리는 {pct(h['by_crowd'][-1]['ratio'])}를 셌다. <b>붐빌수록 머리가 필요하다</b>는 근거.")
    if RUN.get("head_base") and RUN.get("head_p2"):
        rb, rp = RUN["head_base"], RUN["head_p2"]
        out.append(f"직접 수정한 P2 구조: mAP50 {rb['mAP50']:.3f} → {rp['mAP50']:.3f}, "
                   f"mAP50-95 {rb['mAP95']:.3f} → {rp['mAP95']:.3f}로 <b>박스 정확도는 올랐지만</b>, "
                   f"인원 세기(MAE {sm['head_base']['MAE']:.2f} → {sm['head_p2']['MAE']:.2f})와 "
                   f"16px 미만 머리 찾기는 <b>나아지지 않았고</b> 속도는 "
                   f"{sm['speed_ms']['head_base']:.1f} → {sm['speed_ms']['head_p2']:.1f} ms/장으로 느려졌다 (4-2 참고).")
        if SUM15:
            out.append(f"단, 인원 세기용 신뢰도 기준 0.15로 맞추면 <b>P2가 조금 더 정확하게 센다</b> "
                       f"(MAE base {SUM15['head_base']['MAE']:.2f} · P2 {SUM15['head_p2']['MAE']:.2f}, 4-4 참고).")
    if VIDEO:
        m = VIDEO["models"]; best = max((k for k in m if k.startswith("head")), key=lambda k: m[k]["count_ratio"])
        out.append(f"현장 영상(위에서 내려다본 교차로, 화면당 평균 {VIDEO['gt_mean_count']:.0f}명): 전신은 "
                   f"{pct(m['body']['count_ratio'])}~{pct(m['body_1920']['count_ratio'])}만 찾고 붐빌수록 밀도를 <b>더 낮게</b> 봤다"
                   f"(상관 {m['body']['maxd_corr']:.2f}). 머리는 {pct(m[best]['count_ratio'])}를 찾고 붐비는 흐름을 따라갔다(5절).")
    return out


def crowd_points():
    sm = SUMMARY; out = []
    rs = sm["body"]["by_crowd"]
    out.append("전신 YOLO: " + " → ".join(f"{pct(r['ratio'])}" for r in rs) +
               " (사람이 많아질수록 몸이 가려져 박스를 놓치고, 겹친 박스는 하나로 합쳐짐).")
    for k in HEAD_KEYS:
        out.append(f"{NAME[k]}: " + " → ".join(f"{pct(r['ratio'])}" for r in sm[k]["by_crowd"]) +
                   " (머리는 가려지는 일이 적어 붐벼도 비율이 유지됨).")
    return out


def size_points():
    sm = SUMMARY; out = []
    if len(HEAD_KEYS) == 2:
        a, b = (sm[k]["recall_by_size"] for k in HEAD_KEYS)
        for x, y in zip(a, b):
            if x["recall"] is None:
                continue
            d = 100 * (y["recall"] - x["recall"])
            nm = x['range'].replace("0-16px", "16px 미만").replace("64-px", "64px 이상").replace("-", "~")
            out.append(f"{nm}: base {pct(x['recall'])} → P2 {pct(y['recall'])} ({'+' if d >= 0 else ''}{d:.1f}%p)")
        rb, rp = RUN.get("head_base"), RUN.get("head_p2")
        if rb and rp:
            out.append(f"<b>해석</b>: 모든 크기에서 차이가 1%p 이내다. P2를 넣어도 16px 미만 머리는 여전히 절반 정도만 찾았다. "
                       f"P2의 mAP 상승(mAP50-95 +{100 * (rp['mAP95'] - rb['mAP95']):.1f}%p)은 '더 많이 찾기'가 아니라 "
                       "<b>박스 위치를 더 정확히 맞춘 것</b>에서 나왔다(mAP50-95는 IoU 0.5~0.95를 평균해 위치 정확도를 더 많이 반영).")
            out.append("<b>작은 머리가 안 잡히는 이유(추정)</b>: 입력 960으로 줄이면서 원본에서 이미 작던 머리가 몇 픽셀까지 작아진다. "
                       "탐지층을 늘려도 입력에 정보가 없으면 찾을 수 없다. "
                       "그래서 구조 대신 입력을 바꾸는 타일링을 직접 구현해 확인했다(4-5절). 신뢰도 기준 맞추기는 4-4절.")
    return out


def conclusion_points():
    sm = SUMMARY; out = [
        "<b>기존 기능을 다르게 적용</b>: 같은 YOLO 학습 코드에 데이터·클래스·구조를 바꿔 ‘전신’ 대신 ‘머리’를 찾게 했다 "
        "(공지 예시의 ‘마스크 → 썬글라스 추가’와 같은 방식).",
        "<b>직접 이해하고 바꾼 부분</b>: ① CrowdHuman 라벨 변환 규칙(무시 영역 회색 처리) "
        "② Detect 층 80 → 1 클래스와 전이학습 ③ YOLO11 구조 파일에 P2 탐지층 추가."]
    if HEAD_KEYS:
        best = min(HEAD_KEYS, key=lambda k: sm[k]["MAE"])
        out.append(f"<b>왜 머리인가</b>: 붐비는 사진(61명 이상)에서 전신은 {pct(sm['body']['by_crowd'][-1]['ratio'])}, "
                   f"머리는 {pct(sm[best]['by_crowd'][-1]['ratio'])}를 셌다. 인파 사고는 붐빌 때 일어나므로 이 차이가 핵심.")
    if len(HEAD_KEYS) == 2:
        better = RUN["head_p2"]["mAP95"] > RUN["head_base"]["mAP95"] if RUN.get("head_p2") and RUN.get("head_base") else None
        if better is not None:
            out.append("<b>구조 수정의 효과</b>: " + ("P2 구조는 mAP가 올랐고(박스 위치 정확도), 인원 세기용 기준(0.15)에서는 "
                                                "인원 오차도 base보다 조금 작았다. 하지만 16px 미만 머리를 찾는 비율은 그대로였고 속도는 느려졌다. "
                                                "즉 <b>작은 머리 문제는 탐지층 추가만으로는 풀리지 않았다</b>. "
                                                "현장 영상에서는 두 구조의 찾은 비율이 같았으므로(5절) 시연에는 빠른 원본 구조와, "
                                                "그것을 현장 영상으로 추가 학습한 head_ht21을 쓴다."
                                                if better else
                                                "P2 구조가 원본보다 점수가 높지 않았다. 늘어난 계산량만큼 효과가 없었다는 것도 실험 결과로 보고한다."))
    if TILES:
        a, b = TILES["modes"]["full"]["calibrated"], TILES["modes"]["tile2"]["calibrated"]
        out.append(f"<b>가설 → 실패 → 원인 → 해결</b>: 탐지층 추가(P2)로 안 풀린 작은 머리를, 사진을 조각내 확대해 넣는 타일링(직접 구현)으로 "
                   f"16px 미만 재현율 {pct(a['recall_by_size'][0]['recall'])} → {pct(b['recall_by_size'][0]['recall'])}로 올렸다. "
                   "원인은 구조가 아니라 입력 해상도였다. 단, 머리가 작지 않은 현장 영상에서는 효과가 없었다(4-5절).")
    if VIDEO:
        m = VIDEO["models"]
        out.append(f"<b>현장 영상에서도 같은 결론</b>: 전신은 붐빌수록 밀도를 낮게 보는 반대 방향(상관 {m['body']['maxd_corr']:.2f})이라 "
                   "경보에 쓸 수 없고, 머리는 흐름을 따라간다. 다만 머리도 실제보다 적게 세므로 <b>놓치면 위험을 낮게 본다</b> "
                   "→ 경보 기준을 정할 때 이 오차를 감안해야 한다.")
    return out


# ---------- 4-4·5절 ----------
def calib_points():
    a, b = SUMMARY, SUM15; out = []
    for k in HEAD_KEYS:
        out.append(f"{NAME[k]}: MAE {a[k]['MAE']:.2f} → <b>{b[k]['MAE']:.2f}</b>, "
                   f"찾은 비율 {pct(a[k]['found_ratio'])} → {pct(b[k]['found_ratio'])}")
    if len(HEAD_KEYS) == 2:
        win = min(HEAD_KEYS, key=lambda k: b[k]["MAE"])
        lose = [k for k in HEAD_KEYS if k != win][0]
        if a[win]["MAE"] > a[lose]["MAE"]:   # 기준을 바꾸니 순위가 뒤집힌 경우
            out.append(f"<b>P2 결론이 바뀐다</b>: 기본 기준(0.25)에서는 {NAME[lose]}의 인원 오차가 작았지만, 기준을 맞추면 "
                       f"<b>{NAME[win]}</b>가 더 정확하다. 0.25에서는 덜 세는 손해를 더 크게 봤던 것. "
                       "모델 비교는 같은 ‘세기 조건’에서 해야 한다는 교훈.")
        else:
            out.append(f"기준을 맞춰도 {NAME[win]}가 더 정확하다.")
    out.append(f"전신 YOLO는 기준을 낮춰도 {pct(b['body']['found_ratio'])}, 61명 이상 사진에서 "
               f"{pct(b['body']['by_crowd'][-1]['ratio'])}에 그친다. 가려진 몸은 기준을 낮춰도 박스 자체가 잘 나오지 않는다.")
    return out


def tiles_section(W):
    md = TILES["modes"]; TN = {"full": "보통 탐지 (전체 사진 1번)", "tile2": "타일링 2×2 + 전체", "tile3": "타일링 3×3 + 전체"}
    st = [Spacer(1, 8), P("4-5. 타일링: 사진을 조각내 확대해서 탐지 (src/tile_detect.py, 직접 구현)", "h2"),
          P("4-2에서 P2 탐지층을 넣어도 작은 머리 재현율은 그대로였다. 원인을 ‘입력을 960으로 줄이며 작은 머리 정보가 사라짐’으로 추정했으므로, "
            "모델·가중치는 그대로 두고 <b>입력을 바꿨다</b>."),
          B("사진을 n×n 조각으로 나눠(이웃과 20% 겹침) 조각마다 960 크기로 탐지 → 같은 머리가 더 크게 들어간다."),
          B("조각의 안쪽 경계에 닿은 박스는 잘린 머리이므로 버리고, 조각보다 큰 머리를 위해 전체 사진 탐지도 한 번 더한다."),
          B("여러 조각·전체 탐지에서 중복으로 잡힌 머리는 겹친 박스 합치기(NMS, IoU 0.5)를 직접 짜서 하나만 남긴다."),
          Spacer(1, 4)]
    rows = [["방식", "신뢰도<br/>기준", "인원 오차<br/>(MAE)", "맞힌 머리<br/>(재현율)", "16px 미만", "16~32px", "32px 이상", "시간<br/>(ms/장)"]]
    for k, r in md.items():
        s = r["calibrated"]; rs = s["recall_by_size"]
        big = sum(x["found"] for x in rs[2:]) / max(sum(x["total"] for x in rs[2:]), 1)
        rows.append([TN[k], f"{s['conf']:.2f}", f"{s['MAE']:.2f}", pct(s["recall"]), pct(rs[0]["recall"]), pct(rs[1]["recall"]), pct(big),
                     f"{r['ms_per_image']:.0f}"])
    st.append(table(rows, [W * .27, W * .09, W * .11, W * .12, W * .11, W * .10, W * .11, W * .09]))
    st.append(P(f"CrowdHuman 검증 {TILES['images']:,}장, 머리 원본 구조(head_base). 타일링은 박스를 더 많이 내므로 방식마다 ‘센 인원 합 = 실제 인원 합’이 되는 "
                "기준을 따로 맞춘 뒤 비교했다(같은 개수를 셀 때 진짜 머리를 얼마나 맞혔는지). 시간은 사진 읽기 포함이라 1쪽 표의 속도(추론만)와 다르다.", "cap"))
    a, b = md["full"]["calibrated"], md["tile2"]["calibrated"]
    st += [B(f"<b>작은 머리가 잡힌다</b>: 16px 미만 재현율 {pct(a['recall_by_size'][0]['recall'])} → "
             f"<b>{pct(b['recall_by_size'][0]['recall'])}</b>, 전체 재현율 {pct(a['recall'])} → {pct(b['recall'])}, 인원 오차 {a['MAE']:.2f} → {b['MAE']:.2f}. "
             "P2(구조 변경)로 안 되던 것이 입력 변경으로 됐다 → <b>원인은 탐지층이 아니라 입력 해상도</b>."),
           B(f"2×2면 충분하다. 3×3은 재현율이 거의 같고 시간만 늘어난다({md['tile2']['ms_per_image']:.0f} → {md['tile3']['ms_per_image']:.0f} ms/장). "
             f"대가는 속도: 보통 탐지의 약 {md['tile2']['ms_per_image'] / md['full']['ms_per_image']:.1f}배.")]
    if TILES_V:
        mv = TILES_V["models"]
        f = lambda k: " → ".join(pct(mv[k][x]["count_ratio"]) for x in ("full", "tile2", "tile3"))
        st.append(B(f"<b>현장 영상(HT21-03)에서는 효과가 없었다</b>: 찾은 비율(보통 → 2×2 → 3×3)이 원본 구조 {f('head_base')}, "
                    f"현장 추가 학습 {f('head_ht21')}. 이 영상의 머리는 높이 26~45px(중앙 33px)로 작지 않다. "
                    "놓치는 원인이 크기가 아니라 모양(검은 뒤통수 + 검은 외투)이라 확대해도 소용없고, 3×3은 너무 확대돼 헛박스가 늘었다."))
        st.append(B("결론: 타일링은 머리가 작게 찍히는 먼 카메라용 도구다. 시연 영상은 보통 탐지를 그대로 쓴다. "
                    f"(이 실험은 기준을 0.01 단위로 찾아 head_ht21 기준이 {mv['head_ht21']['full']['conf']:.2f}이고, 5절은 0.05 단위로 찾은 0.20이라 찾은 비율이 조금 다르다.)"))
    return st


def fuse_section(W):
    md = FUSE["modes"]; FN = {"body": "전신만 (기존 X)", "head": "머리만 (Y①)", "fuse": "전신 + 머리 결합"}
    st = [PageBreak(), P("4-6. 전신 + 머리 결합 세기 (src/fuse_count.py, 직접 구현)", "h2"),
          P("전신은 한산할 때, 머리는 붐빌 때 강하다. 둘을 합치면 서로 놓친 사람을 채울 수 있는지 확인했다. "
            "그냥 더하면 같은 사람을 두 번 세므로 ‘이 머리가 이 몸의 머리인가’를 판단해 짝지어야 한다."),
          B("짝 규칙: 머리 박스 중심이 전신 박스의 가로 범위 안 + 위쪽 1/3 안이면 후보. 붐비면 한 머리가 여러 전신 박스에 들어가므로, "
            "‘머리 중심 ↔ 전신 박스 윗변 가운데’ 거리가 가까운 짝부터 1:1로 확정한다."),
          B("인원 = 짝 + 머리만 잡힌 사람 + 전신만 잡힌 사람."),
          Spacer(1, 4)]
    rows = [["방식", "머리 / 전신<br/>기준", "인원 오차<br/>(MAE)", "맞힌 사람<br/>(재현율)", "정밀도", "61명 이상<br/>찾은 비율", "짝 / 머리만 / 전신만"]]
    for k in ("body", "head", "fuse"):
        s = md[k]; p = s["parts"]
        rows.append([FN[k], f"{'-' if k == 'body' else format(s['head_conf'], '.2f')} / {s['body_conf'] if s['body_conf'] is not None else '-'}",
                     f"{s['MAE']:.2f}", pct(s["recall"]), pct(s["precision"]), pct(s["by_crowd"][-1]["ratio"]),
                     f"{p['pair']:,} / {p['head_only']:,} / {p['body_only']:,}"])
    st.append(table(rows, [W * .19, W * .12, W * .11, W * .12, W * .10, W * .12, W * .24]))
    st.append(P(f"CrowdHuman 검증 {FUSE['images']:,}장. 결합은 항상 머리만보다 많이 세므로, ‘센 합 = 실제 합’이 되게 머리 기준을 다시 맞췄다"
                f"(전신 기준 0.3·0.5·0.7 중 오차가 가장 작은 {FUSE['best_body_conf']}). 맞힌 사람: 머리 박스가 정답과 IoU 0.5 이상, "
                "전신만 잡힌 경우는 아직 안 맞힌 정답 머리가 전신 박스 위쪽 1/3 안에 있으면 맞힘.", "cap"))
    h, f = md["head"], md["fuse"]
    st += [B(f"<b>결합해도 나아지지 않았다</b>: 인원 오차 {h['MAE']:.2f} → {f['MAE']:.2f}, 재현율 {pct(h['recall'])} → {pct(f['recall'])}. "
             f"전신만 잡혀 추가된 사람은 {f['parts']['body_only']:,}명(전체의 {100 * f['parts']['body_only'] / (f['parts']['pair'] + f['parts']['head_only'] + f['parts']['body_only']):.1f}%)뿐이다."),
           B("<b>이유</b>: 전신이 찾는 사람은 머리 모델도 이미 찾는다. 검증 300장에서 ‘전신만’ 박스 202개를 정답과 대조하니 "
             "머리 모델이 놓친 진짜 사람은 18개(9%)였고, 45개는 이미 머리로 센 사람(머리가 숙여지거나 박스가 잘려 위쪽 1/3 밖 → 중복), "
             "139개는 위쪽에 정답 머리가 없는 박스(머리가 화면 밖·가려짐, 무시 영역, 헛박스)였다."),
           B("즉 전신 박스는 새 사람보다 <b>중복과 헛것을 더 많이</b> 가져온다. 전신 기준을 높일수록(0.3 → 0.7) 결합 결과가 머리만에 가까워지는 것도 같은 뜻이다.")]
    if FUSE_V:
        mv = FUSE_V["models"]; a, b = mv["head_ht21"]["head"], mv["head_ht21"]["fuse"]
        st.append(B(f"현장 영상(HT21-03, 전신은 유리하게 입력 1920): 찾은 비율 {pct(a['found_ratio'])} → {pct(b['found_ratio'])}, "
                    f"전신만 추가는 1,000프레임 합쳐 {b['parts']['body_only']}명. 위에서 본 밀집 장면에서는 전신이 거의 안 잡혀 보탤 것이 없다."))
        st.append(B(f"참고: 이 영상에서 박스 위치까지 맞힌 비율(재현율)은 {pct(a['recall'])}, 정밀도 {pct(a['precision'])}다. "
                    "‘찾은 비율’(개수 비교)에는 헛박스도 들어 있으므로, 실제로 놓친 사람은 개수 차이보다 많다."))
    st.append(B("<b>결론</b>: 인원 세기는 머리만으로 한다. ‘X에 Y를 더한다’가 아니라 <b>‘붐비는 장면에서는 Y(머리)가 X(전신)를 대신한다’</b>는 것이 실험 결과다."))
    return st


VNAME = {"body": "전신 YOLO (입력 960)", "body_1920": "전신 YOLO (입력 1920)", "head_base": "머리 · 원본 구조",
         "head_p2": "머리 · P2 구조", "head_ht21": "머리 · 현장 영상 추가 학습"}


def video_section(W):
    v = VIDEO; m = v["models"]; st = []
    st += [PageBreak(), P("5. 현장 영상 평가 (HT21-03, 학습에 쓰지 않은 영상)", "h1"),
           P("CrowdHuman은 대부분 눈높이 사진이다. 실제 CCTV처럼 <b>위에서 내려다본 영상</b>에서도 되는지 "
             "HT21(CroHD, MOTChallenge 머리 추적 데이터셋, CC BY-NC-SA 3.0)의 HT21-03(시부야 스크램블 교차로, "
             f"{v['frames']}프레임 = 40초, 화면당 평균 {v['gt_mean_count']:.0f}명)으로 평가했다. 모든 프레임에 머리 박스 정답이 있다."),
           B(f"바닥: 화면의 네 점 → {v['floor_m'][0]:.0f}m × {v['floor_m'][1]:.0f}m 직사각형으로 원근 변환(호모그래피). "
             "넓이는 횡단보도 줄 간격 0.9m 가정 + 맨홀 타원 비율로 추정."),
           B("원근 변환은 점 4쌍 → 식 8개 연립방정식을 푸는 코드를 직접 구현했다(src/homography.py). "
             "OpenCV 함수와 행렬 차이 10<super>-14</super> 이하, 점 변환 차이 0.002mm로 같은 답."),
           B("밀도: 머리 박스 아래로 머리 높이×6 지점을 발 위치로 보고 바닥 좌표로 옮긴 뒤, 2m×2m 칸마다 ㎡당 인원. "
             "칸별로 1초 평균(2m 칸은 1명 차이가 ㎡당 0.25명이라 프레임마다 출렁임)."),
           B("정답(gt.txt)의 머리 박스도 똑같은 방법으로 밀도를 계산해 비교했다."),
           B("head_ht21: head_base를 HT21-01·02·04(같은 데이터셋의 <b>다른</b> 장소)로 20 epoch 추가 학습(5분). "
             "HT21-03은 학습에도, 신뢰도 기준(0.20, HT21-02 뒷부분으로 정함)에도 쓰지 않았다."),
           Spacer(1, 4)]
    rows = [["모델", "신뢰도<br/>기준", "찾은 비율", "인원 오차<br/>(명/프레임)", "최대 칸 밀도<br/>치우침 (명/㎡)",
             "밀도 흐름 일치<br/>(상관계수)"]]
    for k, r in m.items():
        c = f"{r['maxd_corr']:.2f}"
        rows.append([VNAME.get(k, k), f"{v['conf'][k]:.2f}", pct(r["count_ratio"]), f"{r['count_MAE']:.0f}",
                     f"{r['maxd_bias']:+.2f}", f"<b>{c}</b>" if r["maxd_corr"] < 0 else c])
    st.append(table(rows, [W * .27, W * .10, W * .12, W * .15, W * .17, W * .19]))
    st.append(P("찾은 비율: 센 인원 합 ÷ 정답 인원 합. 최대 칸 밀도 치우침: 가장 붐비는 칸의 밀도(모델 - 정답)의 평균, "
                "음수면 덜 붐비게 봄. 밀도 흐름 일치: 40초 동안 ‘가장 붐비는 칸 밀도’ 곡선이 정답 곡선과 같이 오르내리는 정도"
                "(1이면 완전히 같고, 음수면 반대로 움직임).", "cap"))
    st += [Image(str(ROOT / "results/eval_video/HT21-03_count.png"), W, W * 640 / 1440),
           P("프레임마다 센 인원. 사람들이 횡단보도로 몰려들수록 머리 모델은 따라 늘지만 전신은 오히려 줄어든다.", "cap"),
           Image(str(ROOT / "results/eval_video/HT21-03_density.png"), W, W * 640 / 1440),
           P("가장 붐비는 2m 칸의 ㎡당 인원(1초 평균). 가로줄은 시연용 경보 기준.", "cap")]
    pts = [f"<b>전신은 붐빌수록 반대로 본다</b>: 상관 {m['body']['maxd_corr']:.2f}(입력 960) / "
           f"{m['body_1920']['maxd_corr']:.2f}(입력 1920). 위에서 내려다보면 몸이 서로 가려 사람이 몰릴수록 박스가 줄어든다. "
           f"입력을 키워도 찾은 비율은 {pct(m['body_1920']['count_ratio'])}뿐. "
           "<b>경보에 쓰면 가장 위험할 때 ‘괜찮다’고 한다.</b>",
           f"머리 모델은 붐비는 흐름을 따라간다(상관 {m['head_base']['maxd_corr']:.2f}). 다만 처음 보는 환경이라 "
           f"{pct(m['head_base']['count_ratio'])}만 찾았고, 놓친 것은 주로 <b>뒤돌아선 검은 머리 + 검은 외투</b>"
           "(배경과 구분이 안 됨)였다."]
    if "head_ht21" in m:
        pts.append(f"현장 영상으로 5분 추가 학습(head_ht21)하니 찾은 비율 {pct(m['head_base']['count_ratio'])} → "
                   f"<b>{pct(m['head_ht21']['count_ratio'])}</b>, 인원 오차 {m['head_base']['count_MAE']:.0f} → "
                   f"{m['head_ht21']['count_MAE']:.0f}명. 같은 학습 코드로 현장에 맞출 수 있다.")
    if "head_p2" in m:
        pts.append(f"P2 구조는 이 영상에서 base와 찾은 비율이 거의 같았다({pct(m['head_base']['count_ratio'])} vs "
                   f"{pct(m['head_p2']['count_ratio'])}).")
    st += [P("결과 해석", "h2")] + [B(t) for t in pts]

    lv = v["levels"]; demo = lv["시연"]
    st += [PageBreak(), P("경보 비교", "h2"),
           B(f"서울시 기준(㎡당 3·4·5명): 정답 밀도가 최대 {v['gt_max_density']:.2f}명/㎡(1초 평균)라 <b>40초 내내 ‘정상’</b>. "
             "모든 모델도 정상으로 판단했다(일치율 100%). 실제로 위험한 장면은 아니었다."),
           B("경보가 동작하는 모습을 보이기 위해 <b>시연용 기준</b>(서울시의 1/3: ㎡당 "
             + " · ".join(f"{t:g}" for t in demo["thresholds"]) + "명)으로 같은 영상을 다시 판단했다. "
             "시연 화면에도 ‘시연용 기준’을 표시한다."),
           Spacer(1, 3)]
    fmt = lambda a: " → ".join(f"{e[0]}초 {e[1]}" for e in a) or "경보 없음"
    rows = [["", "시연 기준 경보 (같은 단계가 5초 이어지면 울림)"], ["정답 머리 박스", fmt(demo["gt_alarms"])]]
    for k in m:
        rows.append([VNAME.get(k, k), fmt(m[k].get("alarms_시연", []))])
    st.append(table(rows, [W * .27, W * .73]))
    st.append(P("시간은 영상 시작부터 초. ‘해제’는 모든 칸이 기준 아래로 5초 이어진 시각.", "cap"))
    st += [B("전신 모델은 시연 기준에서도 <b>경보를 한 번도 울리지 못했다</b>. 머리 모델은 경보를 울리지만 정답보다 늦고 "
             "낮은 단계(주의)에 그쳤다 → 사람을 놓친 만큼 밀도를 실제의 약 0.65배로 봤기 때문."),
           B("시연 영상 만들기: <font size=8>python src/crowd_alarm.py data/ht21/HT21/train/HT21-03/img1 "
             "results/demo_ht21_03.mp4 --model head_ht21 --conf 0.2 --levels 1,1.33,1.67</font> "
             "(영상은 데이터 라이선스 때문에 GitHub에 올리지 않음)")]
    return st


if __name__ == "__main__":
    build()
