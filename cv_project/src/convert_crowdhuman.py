"""CrowdHuman 라벨(.odgt) → YOLO 머리 라벨 변환 (추가 기능 Y①의 데이터 준비).

usage: python src/convert_crowdhuman.py <annotation.odgt> <이미지 폴더> <subset> [시작:끝]
  예) python src/convert_crowdhuman.py data/crowdhuman/annotation_val.odgt data/crowdhuman/Images train 0:4000
      python src/convert_crowdhuman.py data/crowdhuman/annotation_val.odgt data/crowdhuman/Images val 4000:
결과: data/head_yolo/images/<subset>/*.jpg, data/head_yolo/labels/<subset>/*.txt, data/head_yolo/head.yaml

[.odgt 형식] 한 줄 = 사진 한 장(JSON)
  {"ID": 사진이름, "gtboxes": [{"tag": "person"|"mask", "hbox": [x,y,w,h], "fbox": 전신, "vbox": 보이는 부분,
                               "head_attr": {"ignore": 0|1, ...}, "extra": {"ignore": 0|1, ...}}, ...]}
  → 우리는 사람마다 있는 hbox(머리 박스)만 쓴다.

[YOLO 라벨 형식] 사진 한 장 = txt 한 개, 한 줄 = 물체 하나
  "클래스 cx cy w h" (모두 0~1 비율, cx·cy는 박스 중심). 클래스는 0 = head 하나뿐.

[제외 규칙과 이유]
  1) tag == "mask": 사람이 아닌 것(포스터·마네킹) 또는 너무 빽빽해 라벨을 못 단 군중 영역.
  2) head_attr.ignore == 1 또는 extra.ignore == 1: 머리 위치가 불확실하거나 무시하라고 표시된 사람.
  → 그냥 빼기만 하면 그 안의 진짜 머리가 '배경(머리 아님)'으로 학습되어 모델이 헷갈린다.
    Ultralytics에는 '무시 영역' 기능이 없으므로, 그 영역을 회색(114, YOLO 패딩색)으로 칠해 사진에서 지운다.
    칠한 영역에 중심이 들어간 정상 머리 라벨도 함께 뺀다(지워진 머리를 정답으로 두지 않기 위해).
  3) 사진 밖으로 나간 박스는 사진 경계에 맞춰 자르고, 잘린 뒤 2픽셀보다 작으면 뺀다.
"""
import sys, json, shutil, pathlib as _pl
import cv2
from cvio import imread, imwrite   # 한글 경로에서도 되는 사진 읽기·쓰기 (cv2.imread/imwrite 대신)

ROOT = _pl.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "head_yolo"        # CrowdHuman 파생물 → data/ 는 .gitignore 대상(재배포 금지)
GRAY = (114, 114, 114)                   # Ultralytics가 패딩에 쓰는 색 = '아무것도 없음'
MIN_SIZE = 2                             # 이보다 작은 박스는 라벨로 의미 없음


def clip(x, y, w, h, W, H):
    """박스를 사진 안으로 자른다. 반환: (x1, y1, x2, y2) 또는 너무 작으면 None"""
    x1, y1, x2, y2 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x2 - x1 < MIN_SIZE or y2 - y1 < MIN_SIZE:
        return None
    return x1, y1, x2, y2


def convert_one(rec, img_dir, subset):
    src = img_dir / (rec["ID"] + ".jpg")
    img = imread(str(src))
    if img is None:
        return None
    H, W = img.shape[:2]

    heads, ignore_regions = [], []
    for b in rec["gtboxes"]:
        if b["tag"] == "mask":                                   # 규칙 1
            ignore_regions.append(b["fbox"])
            continue
        if b.get("head_attr", {}).get("ignore") == 1 or b.get("extra", {}).get("ignore") == 1:
            ignore_regions.append(b["hbox"])                     # 규칙 2: 그 머리 자리만 지움
            continue
        heads.append(b["hbox"])

    # 무시 영역을 회색으로 칠하기
    painted = []
    for r in ignore_regions:
        c = clip(*r, W, H)
        if c:
            cv2.rectangle(img, (c[0], c[1]), (c[2], c[3]), GRAY, -1)
            painted.append(c)

    def in_painted(cx, cy):
        return any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in painted)

    lines = []
    for h in heads:
        c = clip(*h, W, H)                                       # 규칙 3
        if not c:
            continue
        x1, y1, x2, y2 = c
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if in_painted(cx, cy):                                   # 칠해서 지워진 머리는 정답에서도 뺌
            continue
        # 픽셀 좌표 → 0~1 비율 (YOLO 형식)
        lines.append(f"0 {cx / W:.6f} {cy / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")

    dst = OUT / "images" / subset / src.name
    if painted:
        imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        shutil.copyfile(src, dst)                                # 칠할 게 없으면 원본 그대로(화질 유지)
    (OUT / "labels" / subset / (src.stem + ".txt")).write_text("\n".join(lines))
    return len(lines), len(painted)


def main():
    odgt, img_dir, subset = sys.argv[1], _pl.Path(sys.argv[2]), sys.argv[3]
    a, b = (sys.argv[4].split(":") + [""])[:2] if len(sys.argv) > 4 else ("", "")
    recs = [json.loads(l) for l in open(odgt, encoding="utf-8")]
    recs = recs[int(a or 0): int(b) if b else None]

    for d in ("images", "labels"):
        (OUT / d / subset).mkdir(parents=True, exist_ok=True)

    n_img = n_head = n_paint = missing = 0
    for i, rec in enumerate(recs):
        r = convert_one(rec, img_dir, subset)
        if r is None:
            missing += 1
            continue
        n_img += 1; n_head += r[0]; n_paint += r[1]
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(recs)}")

    # 학습 설정 파일: 경로와 클래스 이름 (nc=1 → Detect 층 출력이 80 클래스에서 1 클래스로 바뀜)
    (OUT / "head.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: head\n")
    print(f"[{subset}] 사진 {n_img}장, 머리 {n_head}개, 회색 처리 영역 {n_paint}개, 이미지 없음 {missing}장")
    print("설정 파일:", OUT / "head.yaml")


if __name__ == "__main__":
    main()
