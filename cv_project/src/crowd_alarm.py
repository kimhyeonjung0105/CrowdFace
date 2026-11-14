"""CrowdFace 최종 시연: 우리가 학습한 머리 YOLO + 바닥 4점 원근 변환 + 2m 칸 밀도 + 5초 경보.

usage:
  python src/crowd_alarm.py <입력 사진|영상|프레임 사진 폴더> <출력 사진|영상> [옵션]
  예) python src/crowd_alarm.py data/street.mp4 results/alarm.mp4
      python src/crowd_alarm.py in.jpg out.jpg --floor 330,470,1380,470,2300,1271,-600,1271 --size 12,20
옵션
  --floor x1,y1,...,x4,y4  바닥 위 직사각형의 네 꼭짓점(사진 좌표). 순서: 먼 왼쪽 → 먼 오른쪽 → 가까운 오른쪽 → 가까운 왼쪽
  --size 가로,세로          그 직사각형의 실제 크기(m). 가로 = 1→2 변, 세로 = 2→3 변
                            둘 다 없으면 창이 떠서 바닥 네 점을 클릭하고, 크기는 터미널에 입력 → <입력>.floor.json 에 저장해 다음부터 재사용
  --model head_base         models/ 안의 머리 가중치 이름 (P2 구조는 head_p2)
  --imgsz 960               YOLO 입력 크기 (학습 크기 960, 사람이 작게 찍히면 1280 이상)
  --foot 6                  머리 → 발 위치 보정(머리 박스 높이의 몇 배 아래가 발인지). 바로 위에서 찍으면 0
  --max-sec N               영상 앞부분 N초만 처리 (시험용)
  --levels 3,4,5            주의·경계·심각 기준(서울시). 실제 영상이 기준에 못 미칠 때 경보 동작을 보이려면
                            낮춘 기준(예: 1.5,2,2.5)을 쓰고, 화면에 '시연용 기준'으로 표시된다

기존 crowd_final.py와 무엇이 다른가 (추가 기능 Y①·Y② 적용)
  1) 머리: P2PNet(남의 공개 모델) → 우리가 CrowdHuman으로 직접 학습한 head_base.pt
  2) 바닥 넓이: 사람 키 1.65m·화각 60° 가정 → 바닥 네 점 원근 변환(호모그래피)으로 실제 m 단위 계산
  3) 경보: 사진 한 장의 단계 → 영상에서 같은 단계 이상이 5초 이어질 때 경보, 시각·인원 CSV 기록
"""
import os, sys, json, csv, argparse, pathlib as _pl
from collections import deque
import numpy as np, cv2
from cvio import imread, imwrite   # 한글 경로에서도 되는 사진 읽기·쓰기 (cv2.imread/imwrite 대신)
from PIL import Image, ImageDraw, ImageFont
ROOT = _pl.Path(__file__).resolve().parent.parent
from ultralytics import YOLO

CELL = 2.0                                     # 칸 한 변 2m (계획서 기준)
LEVELS = [(5, "심각", (220, 40, 40)), (4, "경계", (255, 140, 0)), (3, "주의", (240, 200, 0))]   # ㎡당 인원, 서울시 기준
SEOUL = (3, 4, 5)
LEVEL_NOTE = ["기준: 서울시 인파 관리 (㎡당 3·4·5명)"]


def set_levels(ths):
    """주의·경계·심각 기준 바꾸기 (시연 모드). 실제 영상이 기준에 못 미치면 경보가 안 울리므로,
    경보 로직이 동작하는 모습을 보여 줄 때만 낮춘 기준을 쓰고 화면에 '시연용 기준'이라고 표시한다."""
    t = sorted(ths)
    LEVELS[:] = [(t[2], "심각", LEVELS[0][2]), (t[1], "경계", LEVELS[1][2]), (t[0], "주의", LEVELS[2][2])]
    if tuple(t) != SEOUL:
        LEVEL_NOTE[:] = [f"시연용 기준 (㎡당 {t[0]:g}·{t[1]:g}·{t[2]:g}명)", "실제 서울시 기준은 3·4·5명"]
NORMAL = (0, "정상", (60, 170, 90))
HOLD = 5.0                                     # 같은 단계 이상이 몇 초 이어지면 경보
COUNT_CONF = 0.15                              # 인원 세기용 신뢰도 기준 (아래 main의 설명 참고)
MIN_AREA = 2.0                                # 이보다 좁은 가장자리 칸은 계산 안 함 (작은 칸은 한두 명만 있어도 가짜 고밀도)


def font(bold):
    for p in (["C:/Windows/Fonts/malgunbd.ttf"] if bold else []) + ["C:/Windows/Fonts/malgun.ttf",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-%s.ttc" % ("Bold" if bold else "Regular")]:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("한글 글꼴(맑은 고딕)을 찾지 못했습니다")
FB, FR = font(True), font(False)
F = lambda s: ImageFont.truetype(FB, s)
R = lambda s: ImageFont.truetype(FR, s)


def level_of(d):
    """㎡당 인원 → (기준, 이름, 색)"""
    return next((lv for lv in LEVELS if d >= lv[0]), NORMAL)


# ---------------- 바닥 설정 (원근 변환) ----------------
def ask_floor(frame, cfg_path):
    """창에서 바닥 네 점을 클릭받고 실제 크기를 입력받아 저장"""
    pts, win = [], "바닥 네 점 클릭: 먼 왼쪽 > 먼 오른쪽 > 가까운 오른쪽 > 가까운 왼쪽 (Enter 완료, R 다시)"
    scale = min(1.0, 1400 / frame.shape[1])
    def on_click(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))
    cv2.namedWindow(win); cv2.setMouseCallback(win, on_click)
    while True:
        v = cv2.resize(frame, None, fx=scale, fy=scale)
        for i, p in enumerate(pts):
            cv2.circle(v, (int(p[0] * scale), int(p[1] * scale)), 6, (0, 255, 255), -1)
            cv2.putText(v, str(i + 1), (int(p[0] * scale) + 8, int(p[1] * scale) - 8), 0, 0.8, (0, 255, 255), 2)
        if len(pts) > 1:
            cv2.polylines(v, [(np.array(pts) * scale).astype(np.int32)], len(pts) == 4, (0, 255, 255), 2)
        cv2.imshow(win, v); k = cv2.waitKey(30)
        if k in (ord("r"), ord("R")):
            pts.clear()
        if k == 13 and len(pts) == 4:
            break
    cv2.destroyAllWindows()
    w = float(input("1→2 변의 실제 길이(m): ")); d = float(input("2→3 변의 실제 길이(m): "))
    cfg = {"floor": [list(map(float, p)) for p in pts], "size": [w, d]}
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")
    print("저장:", cfg_path)
    return cfg


from homography import find_homography, apply_homography   # 원근 변환 직접 구현 (cv2.getPerspectiveTransform 대신)


class Floor:
    """사진 좌표 ↔ 바닥 좌표(m). 바닥은 평면이므로 네 점 대응만으로 원근 변환 행렬(호모그래피) 3×3이 정해진다."""
    def __init__(self, pts, size):
        self.w, self.d = size
        self.quad = np.float32(pts)
        world = np.float32([[0, 0], [self.w, 0], [self.w, self.d], [0, self.d]])
        self.H = find_homography(self.quad, world)                    # 사진 → 바닥(m). 직접 구현(src/homography.py), OpenCV와 같은 답
        self.Hi = np.linalg.inv(self.H)                                # 바닥(m) → 사진 (칸 그리기용)
        nx, ny = int(np.ceil(self.w / CELL - 1e-6)), int(np.ceil(self.d / CELL - 1e-6))
        self.cells = []                                                # (i, j, 넓이㎡, 사진 위 꼭짓점 4개)
        for i in range(nx):
            for j in range(ny):
                x0, y0 = i * CELL, j * CELL
                x1, y1 = min(x0 + CELL, self.w), min(y0 + CELL, self.d)
                area = (x1 - x0) * (y1 - y0)
                if area < MIN_AREA:
                    continue
                c = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).reshape(-1, 1, 2)
                self.cells.append((i, j, area, apply_homography(self.Hi, c).astype(np.float32)))

    def to_floor(self, xy):
        if len(xy) == 0:
            return np.zeros((0, 2), np.float32)
        return apply_homography(self.H, xy)

    def density(self, feet):
        """발 위치(사진 좌표) → 칸별 ㎡당 인원, 구역 안 인원"""
        g = self.to_floor(feet)
        inside = (g[:, 0] >= 0) & (g[:, 0] < self.w) & (g[:, 1] >= 0) & (g[:, 1] < self.d)
        cnt = {}
        for x, y in g[inside]:
            k = (int(x // CELL), int(y // CELL)); cnt[k] = cnt.get(k, 0) + 1
        return [(cnt.get((i, j), 0) / a, cnt.get((i, j), 0), poly) for i, j, a, poly in self.cells], int(inside.sum())


# ---------------- 경보 (시간 조건) ----------------
class Smoother:
    """칸별 ㎡당 인원을 최근 SMOOTH초 평균. 2m 칸은 한 명 차이가 ㎡당 0.25명이라, 프레임마다 한두 명을 놓치거나
    사람이 칸 경계를 넘나들면 값이 크게 출렁인다. 그러면 '5초 연속' 조건이 계속 초기화돼 경보가 안 울린다."""
    def __init__(self, n):
        self.buf = deque(maxlen=max(1, int(n)))

    def __call__(self, dens):
        self.buf.append(np.asarray(dens, float))
        return np.mean(self.buf, axis=0)


SMOOTH = 1.0   # 초


class Alarm:
    """단계마다 '그 단계 이상'이 시작된 시각을 기억 → HOLD초 이상 이어진 가장 높은 단계가 경보.
    한 프레임만 튀는 값으로 경보가 울리거나, 경계↔심각을 오가며 시간이 초기화되는 것을 막는다."""
    def __init__(self, log_path=None):
        self.since = {th: None for th, *_ in LEVELS}
        self.active = None; self.rows = []; self.log_path = log_path

    def update(self, t, maxd, n):
        for th, *_ in LEVELS:
            if maxd >= th:
                self.since[th] = t if self.since[th] is None else self.since[th]
            else:
                self.since[th] = None
        on = next((lv for lv in LEVELS if self.since[lv[0]] is not None and t - self.since[lv[0]] >= HOLD), None)
        if (on and on[1]) != (self.active and self.active[1]):           # 경보 단계가 바뀐 순간만 기록
            self.rows.append([f"{t:.1f}", on[1] if on else "해제", f"{maxd:.2f}", n])
            print(f"  {t:6.1f}초  경보 {'→ ' + on[1] if on else '해제'}  (최대 ㎡당 {maxd:.2f}명, 구역 안 {n}명)")
        self.active = on
        cur = level_of(maxd)
        held = t - self.since[cur[0]] if cur[0] and self.since[cur[0]] is not None else 0.0
        return on, cur, held

    def save(self):
        if self.log_path:
            with open(self.log_path, "w", newline="", encoding="utf-8-sig") as f:      # 엑셀에서 한글이 안 깨지게 BOM
                w = csv.writer(f); w.writerow(["시각(초)", "경보", "최대 ㎡당 인원", "구역 안 인원"]); w.writerows(self.rows)
            print("경보 기록:", self.log_path, f"({len(self.rows)}건)")


# ---------------- 한 프레임 처리·그리기 ----------------
PANEL = 380


def process(frame, model, floor, args):
    # max_det: Ultralytics 기본은 한 장에 300개까지 → 초밀집 장면(정답 338명 프레임도 있음)에서 잘리므로 늘림
    r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, max_det=1000, device=args.device, verbose=False)[0]
    b = r.boxes.xyxy.cpu().numpy()
    # 바닥 원근 변환은 '바닥 위의 점'에만 맞다 → 머리 대신 발 위치를 쓴다.
    # 발은 가려져 안 보이므로 머리 박스 아래로 (머리 높이 × foot)만큼 내려 추정 (사람 키 ≈ 머리 7개).
    feet = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3] + args.foot * (b[:, 3] - b[:, 1])], 1) if len(b) else np.zeros((0, 2))
    cells, n_in = floor.density(feet)
    return b, cells, n_in


def draw(frame, b, cells, n_in, floor, cur, alarm, held, t, args):
    # 작은 영상(예: 960x540)은 오른쪽 글자판이 다 안 들어가므로 화면용으로만 키워서 그린다 (탐지는 원래 크기로 이미 끝남)
    k = max(1.0, 1000 / frame.shape[0])
    quad = floor.quad
    if k > 1:
        frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
        b = b * k; cells = [(d, n, p * k) for d, n, p in cells]; quad = quad * k
    img = frame.copy(); over = img.copy()
    for d, n, poly in cells:                                   # 칸 색: 정상은 테두리만, 주의 이상은 칠함
        lv = level_of(d)
        if lv[0]:
            cv2.fillPoly(over, [poly.astype(np.int32)], lv[2][::-1])
    img = cv2.addWeighted(over, 0.35, img, 0.65, 0)
    for d, n, poly in cells:
        cv2.polylines(img, [poly.astype(np.int32)], True, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.polylines(img, [quad.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
    for x1, y1, x2, y2 in b.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (52, 104, 235), 2, cv2.LINE_AA)     # 머리 모델 색(보고서와 같은 주황)
    H, W = img.shape[:2]
    canvas = Image.new("RGB", (W + PANEL, H), (17, 30, 50))
    canvas.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (0, 0))
    dr = ImageDraw.Draw(canvas)
    best = max(cells, key=lambda c: c[0]) if cells else None
    if best and best[0] > 0:                                   # 가장 붐빈 칸에 숫자
        p = best[2]; cx = p[:, 0].mean(); ty = p[:, 1].min() - 38
        txt = f"㎡당 {best[0]:.1f}명"; tw = dr.textlength(txt, font=F(22))
        dr.rounded_rectangle([cx - tw / 2 - 8, ty, cx + tw / 2 + 8, ty + 32], 6, fill=(17, 30, 50))
        dr.text((cx - tw / 2, ty + 2), txt, font=F(22), fill=(255, 255, 255))
    x0, y = W + 28, 30
    dr.text((x0, y), "CrowdFace 밀집도 경보", font=F(28), fill=(255, 255, 255)); y += 64
    if t is not None:
        dr.text((x0, y), f"{t:6.1f}초", font=R(18), fill=(150, 165, 185)); y += 34
    for k, v in [("머리 탐지 (화면 전체)", f"{len(b)}명"), ("분석 구역 안", f"{n_in}명"),
                 ("가장 붐빈 2m 칸", f"㎡당 {best[0]:.1f}명" if best else "-")]:
        dr.text((x0, y), k, font=R(20), fill=(170, 185, 205))
        dr.text((x0, y + 26), v, font=F(36), fill=(255, 255, 255)); y += 92
    dr.text((x0, y), "현재 단계", font=R(20), fill=(170, 185, 205))
    dr.rounded_rectangle([x0, y + 30, x0 + 150, y + 86], 12, fill=cur[2])
    dr.text((x0 + 75 - dr.textlength(cur[1], font=F(32)) / 2, y + 36), cur[1], font=F(32), fill=(255, 255, 255))
    if t is not None and cur[0]:
        dr.text((x0 + 166, y + 46), f"{held:.1f}초째", font=R(20), fill=(225, 230, 240))
    y += 104
    if t is not None:                                           # 영상일 때만 시간 조건 경보
        if alarm:
            dr.rounded_rectangle([x0, y, x0 + PANEL - 56, y + 56], 10, fill=alarm[2])
            dr.text((x0 + 14, y + 12), f"경보: {alarm[1]} ({HOLD:.0f}초 이상)", font=F(24), fill=(255, 255, 255))
        else:
            dr.text((x0, y + 14), f"경보 없음 (같은 단계 {HOLD:.0f}초 지속 시 경보)", font=R(18), fill=(150, 165, 185))
        y += 76
    for th, name, c in LEVELS[::-1]:
        dr.rectangle([x0, y + 5, x0 + 18, y + 23], fill=c)
        dr.text((x0 + 28, y), f"{name}  ㎡당 {th:g}명 이상", font=R(19), fill=(225, 230, 240)); y += 30
    note = [f"머리: 직접 학습한 YOLO11n ({args.model})", f"바닥 {floor.w:g}m × {floor.d:g}m, 원근 변환 · 칸 {CELL:g}m",
            *LEVEL_NOTE]
    for i, l in enumerate(note):
        dr.text((x0, H - 28 * len(note) - 12 + i * 28), l, font=R(16), fill=(150, 165, 185))
    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


class FrameDir:
    """프레임 사진 폴더(예: HT21 원본 img1/000001.jpg …)를 영상처럼 읽는다. 초당 프레임은 옆의 seqinfo.ini(frameRate), 없으면 25."""
    def __init__(self, d):
        self.files = sorted(p for p in d.iterdir() if p.suffix.lower() in (".jpg", ".png")); self.i = 0
        ini = d.parent / "seqinfo.ini"
        rate = [l.split("=")[1] for l in ini.read_text().splitlines() if l.startswith("frameRate")] if ini.exists() else []
        self.fps = float(rate[0]) if rate else 25.0

    def read(self):
        if self.i >= len(self.files):
            return False, None
        self.i += 1
        return True, imread(str(self.files[self.i - 1]))

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0

    def set(self, prop, v):
        self.i = int(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--floor"); ap.add_argument("--size")
    ap.add_argument("--model", default="head_base"); ap.add_argument("--imgsz", type=int, default=960)
    # 신뢰도 기준 0.15: CrowdHuman 검증 4,370장에서 '센 인원 합 = 실제 인원 합'(100.4%)이 되는 값 (0.25면 87%로 덜 셈)
    ap.add_argument("--conf", type=float, default=COUNT_CONF); ap.add_argument("--foot", type=float, default=6.0)
    ap.add_argument("--max-sec", type=float); ap.add_argument("--device", default=None)
    ap.add_argument("--levels", default="3,4,5", help="주의,경계,심각 기준(㎡당 명). 시연 모드 예: 1.5,2,2.5")
    a = ap.parse_args()
    set_levels(list(map(float, a.levels.split(","))))
    src, out = _pl.Path(a.src), _pl.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    is_video = src.is_dir() or src.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv")
    cap = (FrameDir(src) if src.is_dir() else cv2.VideoCapture(str(src))) if is_video else None
    first = cap.read()[1] if is_video else imread(str(src))
    if first is None:
        sys.exit(f"입력을 읽지 못했습니다: {src}")

    cfg_path = src.with_name(src.name + ".floor.json")
    if a.floor and a.size:
        cfg = {"floor": np.float32(a.floor.split(",")).reshape(4, 2).tolist(), "size": list(map(float, a.size.split(",")))}
    elif cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")); print("바닥 설정 불러옴:", cfg_path)
    else:
        cfg = ask_floor(first, cfg_path)
    floor = Floor(cfg["floor"], cfg["size"])
    model = YOLO(str(ROOT / "models" / f"{a.model}.pt"))

    if not is_video:
        b, cells, n_in = process(first, model, floor, a)
        maxd = max((c[0] for c in cells), default=0)
        imwrite(str(out), draw(first, b, cells, n_in, floor, level_of(maxd), None, 0, None, a))
        print(f"머리 {len(b)}명, 구역 안 {n_in}명, 최대 ㎡당 {maxd:.2f}명 → {level_of(maxd)[1]}  저장: {out}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vw = None; alarm = Alarm(out.with_suffix(".csv")); i = 0; smooth = Smoother(SMOOTH * fps)
    while True:
        ok, fr = cap.read()
        if not ok or (a.max_sec and i / fps > a.max_sec):
            break
        t = i / fps
        b, cells, n_in = process(fr, model, floor, a)
        sm = smooth([c[0] for c in cells])                          # 칸별 1초 평균 → 표시·단계·경보 모두 이 값
        cells = [(d, n, poly) for d, (_, n, poly) in zip(sm, cells)]
        maxd = float(sm.max()) if len(sm) else 0.0
        on, cur, held = alarm.update(t, maxd, n_in)
        img = draw(fr, b, cells, n_in, floor, cur, on, held, t, a)
        if vw is None:                                             # 첫 화면 크기로 저장기 생성
            H, W = img.shape[0] // 2 * 2, img.shape[1] // 2 * 2    # mp4 코덱은 가로·세로가 짝수여야 함
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        vw.write(img[:H, :W]); i += 1
        if i % int(fps * 5) == 0:
            print(f"  {t:6.1f}초 처리 중 (머리 {len(b)}명, 최대 ㎡당 {maxd:.2f}명)")
    vw.release(); alarm.save()
    print(f"영상 저장: {out} ({i}프레임)")


if __name__ == "__main__":
    main()
