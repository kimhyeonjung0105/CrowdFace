"""머리 탐지 학습 결과 보고서(PDF) 만들기.
먼저 python src/eval_heads.py 를 실행해 results/eval/summary.json 을 만든 뒤 실행.
usage: python src/make_report.py
결과: results/head_detection_report.pdf (그림은 results/report/)
"""
import os, re, csv, json, datetime, pathlib as _pl
import numpy as np, cv2, torch
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
        p = FIG / f"ex_{k}.jpg"; cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
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
    story += [P("머리 탐지 학습 결과 보고서", "title"),
              P(f"군중 밀집도 기반 인파 사고 조기 경보 · 추가 기능 Y① &nbsp;|&nbsp; 202104183 김현중, 202204201 김태섭 &nbsp;|&nbsp; {today}", "sub"),
              P("1. 한눈에 보기", "h1"),
              P("기존 YOLO는 사람 <b>전신</b>을 찾는다. 우리는 같은 YOLO 학습 코드를 이해하고 "
                "<b>데이터·출력 클래스·모델 구조</b>를 바꿔 <b>머리</b>를 찾도록 직접 학습시켰다. "
                f"CrowdHuman 검증 사진 {sm['images']:,}장(실제 인원 {sm['gt_total']:,}명)에서 인원을 세어 비교했다."),
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
                   f"속도: RTX 4070, 입력 960, 추론만({sm.get('speed_note', '')}).", "cap"))
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

    # ----- 5. 정리 -----
    story += [PageBreak(), P("5. 정리와 발표 포인트", "h1")] + [B(t) for t in conclusion_points()]
    story += [P("한계", "h2"),
              B("평가는 CrowdHuman 검증 사진 기준이다. CCTV처럼 위에서 내려다본 사진에서는 따로 확인해야 한다."),
              B("인원은 신뢰도 0.25 이상 박스 개수로 셌다. 기준값에 따라 더 세거나 덜 셀 수 있다."),
              B("P2는 메모리 때문에 batch 8로 학습했다(base는 16). 유효 batch는 64로 같지만 BatchNorm 통계는 작은 batch로 계산된다."),
              B("정답 인원은 무시 영역을 뺀 머리 라벨 개수다. 회색으로 지운 영역의 사람은 정답에도, 예측에도 들어가지 않는다."),
              P("다음 할 일", "h2"),
              B("머리 모델(head_base)을 crowd_final.py에 넣어 P2PNet(임시 대체)을 바꾸기"),
              B("작은 머리(16px 미만, 전체의 30%) 찾기: 입력 크기 키우기 · 사진 조각내 확대 탐지(타일링) 비교"),
              B("전신·머리 결합 세기 (머리 중심이 전신 박스 위쪽 1/3 안이면 같은 사람)"),
              B("바닥 4점 원근 변환으로 ㎡당 밀도 → 경보 (응용)")]

    doc = SimpleDocTemplate(str(PDF), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title="머리 탐지 학습 결과 보고서")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print("PDF:", PDF)


def footer(c, d):
    c.saveState(); c.setFont("Malgun", 7.5); c.setFillColor(colors.HexColor(MUTED))
    c.drawString(18 * mm, 9 * mm, "컴퓨터비전 미니프로젝트 · 머리 탐지 학습 결과")
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
                       "다음 실험은 입력 크기 키우기, 사진을 조각내 확대해서 탐지하기(타일링), 모델별 신뢰도 기준 맞추기.")
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
            out.append("<b>구조 수정의 효과</b>: " + ("P2 구조는 mAP가 올랐지만(박스 위치 정확도), 인원 세기와 작은 머리 찾기는 그대로였고 "
                                                "속도는 느려졌다. 즉 <b>작은 머리 문제는 탐지층 추가만으로는 풀리지 않았다</b>. "
                                                "가설을 세우고 직접 확인해 '안 됐다'까지 보여 주는 것도 실험 결과이고, "
                                                "인원 세기에는 원본 구조(head_base)를 쓴다."
                                                if better else
                                                "P2 구조가 원본보다 점수가 높지 않았다. 늘어난 계산량만큼 효과가 없었다는 것도 실험 결과로 보고한다."))
    return out


if __name__ == "__main__":
    build()
