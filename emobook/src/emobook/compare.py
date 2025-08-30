# Downsampling, DTW similarity, and benchmark shelf I/O
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import json, numpy as np

try:
    from fastdtw import fastdtw
except Exception:
    fastdtw = None

def downsample_series(y: List[float], m: int = 400) -> List[float]:
    if len(y) <= m: return list(map(float, y))
    x = np.linspace(0.0, 1.0, num=len(y))
    xi = np.linspace(0.0, 1.0, num=m)
    yi = np.interp(xi, x, np.array(y, float))
    return yi.tolist()

def dtw_distance(a: List[float], b: List[float]) -> float:
    A, B = np.array(a, float), np.array(b, float)
    if fastdtw is not None:
        dist, _ = fastdtw(A, B); return float(dist)
    n, m = len(A), len(B)
    dp = np.full((n+1, m+1), np.inf); dp[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(A[i-1] - B[j-1])
            dp[i,j] = cost + min(dp[i-1,j], dp[i,j-1], dp[i-1,j-1])
    return float(dp[n,m])

def save_benchmark(title: str, arc: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "title": title,
            "n_chunks": arc.get("n_chunks"),
            "window": arc.get("window"),
            "avg_cov": float(np.mean(arc.get("coverage", []))) if arc.get("coverage") else 0.0,
        },
        "arc": {
            "v": downsample_series(arc["v"], 400),
            "a": downsample_series(arc["a"], 400),
            "d": downsample_series(arc["d"], 400),
            "vad_fused": downsample_series(arc["vad_fused"], 400),
        }
    }
    key = "".join(c if c.isalnum() or c in "._-" else "_" for c in title.lower())
    path = out_dir / f"{key}.vad.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path

def load_benchmarks(dir_path: Path) -> Dict[str, Dict[str, Any]]:
    shelf = {}
    for f in Path(dir_path).glob("*.vad.json"):
        try:
            shelf[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
    return shelf

def compare_against_bench(user_arc: Dict[str, Any], shelf: Dict[str, Dict[str, Any]], topk=5) -> List[Dict[str, Any]]:
    q = downsample_series(user_arc["vad_fused"], 400)
    out = []
    for key, payload in shelf.items():
        dist = dtw_distance(q, payload["arc"]["vad_fused"])
        out.append({"key": key, "title": payload["meta"]["title"], "dist": float(dist),
                    "avg_cov": float(payload["meta"].get("avg_cov", 0.0))})
    out.sort(key=lambda r: r["dist"])
    return out[:topk]
