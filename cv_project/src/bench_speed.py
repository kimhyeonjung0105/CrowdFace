"""모델별 추론 속도를 공정하게 다시 잰다 → results/eval/summary.json 의 speed_ms 를 덮어씀.

usage: python src/bench_speed.py [장수=300] [반복=3]
왜 따로 재나: eval_heads.py는 모델을 하나씩 차례로 돌리므로, 그 사이 다른 프로그램(게임 등)이
GPU를 쓰면 한 모델만 느리게 찍힌다. 여기서는 같은 사진에 세 모델을 번갈아 돌리고
여러 번 반복한 값의 중앙값을 써서 그런 흔들림을 줄인다.
"""
import sys, json, time, pathlib as _pl
import numpy as np, torch
ROOT = _pl.Path(__file__).resolve().parent.parent
from ultralytics import YOLO

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
REP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
IMGSZ = 960
imgs = sorted((ROOT / "data" / "head_yolo" / "images" / "val").glob("*.jpg"))[:N]
paths = {"body": "yolo11n.pt", "head_base": "head_base.pt", "head_p2": "head_p2.pt"}
models = {k: YOLO(str(ROOT / "models" / v)) for k, v in paths.items() if (ROOT / "models" / v).exists()}

for k, m in models.items():                       # GPU 예열
    for p in imgs[:10]:
        m.predict(str(p), imgsz=IMGSZ, device=0, verbose=False)

ms = {k: [] for k in models}
for _ in range(REP):
    for p in imgs:
        for k, m in models.items():               # 같은 사진에 번갈아 → 조건이 같음
            r = m.predict(str(p), imgsz=IMGSZ, conf=0.25, classes=[0] if k == "body" else None,
                          device=0, verbose=False)[0]
            ms[k].append(r.speed["inference"])

res = {k: float(np.median(v)) for k, v in ms.items()}
for k, v in res.items():
    print(f"{k}: {v:.2f} ms/장 (중앙값, {len(ms[k])}회)")
sj = ROOT / "results" / "eval" / "summary.json"
s = json.loads(sj.read_text(encoding="utf-8"))
s["speed_ms"] = res
s["speed_note"] = f"검증 {N}장 × {REP}회, 세 모델 번갈아 측정한 중앙값"
sj.write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
