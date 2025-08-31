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

def summarize_arc(arc: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np
    v = np.array(arc["v"], float); a = np.array(arc["a"], float); d = np.array(arc["d"], float)
    fused = np.array(arc["vad_fused"], float)
    cov = np.array(arc.get("coverage", []), float) if arc.get("coverage") else np.array([], float)

    # z-score helpers
    def z(x):
        m = np.nanmean(x); s = np.nanstd(x) + 1e-9
        return (x - m) / s

    Vz, Az, Dz = z(v), z(a), z(d)
    oppA = ((np.sign(Az) != np.sign(Vz)) & (np.abs(Az) >= 0.8)).mean() if len(Vz) else 0.0
    oppD = ((np.sign(Dz) != np.sign(Vz)) & (np.abs(Dz) >= 0.8)).mean() if len(Vz) else 0.0

    # intensity + extrema
    def count_extrema(y: np.ndarray, min_amp=0.05):
        if len(y) < 3: return 0, 0
        up = y[1:-1] > y[:-2]; dn = y[1:-1] > y[2:]
        peaks_idx = np.where(up & dn)[0] + 1
        up2 = y[1:-1] < y[:-2]; dn2 = y[1:-1] < y[2:]
        troughs_idx = np.where(up2 & dn2)[0] + 1
        def strong(idxs):
            keep = []
            for i in idxs:
                left = y[i] - y[i-1]
                right = y[i] - y[i+1]
                if abs(left) >= min_amp or abs(right) >= min_amp:
                    keep.append(i)
            return len(keep)
        return strong(peaks_idx), strong(troughs_idx)

    peaks_f, troughs_f = count_extrema(fused, 0.05)
    std_v = float(np.std(v)); std_fused = float(np.std(fused))

    # qualitative labels (no numbers leaked to prompt unless we choose)
    if std_fused >= 0.07 or peaks_f >= 20:
        intensity = "rollercoaster — frequent swings and sharp turns"
    elif std_fused >= 0.045 or peaks_f >= 10:
        intensity = "undulating — notable rises and dips without whiplash"
    else:
        intensity = "even-keeled — steady emotional current"

    if oppA >= 0.3:
        a_note = "tension often cuts against tone (calm dread or nervous joy)"
    elif oppA >= 0.15:
        a_note = "tension sometimes pushes against tone"
    else:
        a_note = "tension mostly tracks with tone"

    # dominance role vs valence/arousal variance shares
    var_v, var_a, var_d = float(np.var(v)), float(np.var(a)), float(np.var(d))
    total_var = var_v + var_a + var_d + 1e-12
    share_v = var_v / total_var; share_a = var_a / total_var; share_d = var_d / total_var
    if share_d >= 0.40:
        d_note = "strong sense of agency/control shapes the feel"
    elif share_d >= 0.28:
        d_note = "shifts in agency/control matter throughout"
    else:
        d_note = "agency plays a lighter, background role"

    return {
        "n_chunks": int(len(v)),
        "window": int(arc.get("window", 0)),
        "avg_cov": float(np.nanmean(cov)) if cov.size else 0.0,
        "intensity": intensity,
        "a_note": a_note,
        "d_note": d_note,
        "std_v": std_v,
        "std_fused": std_fused,
        "peaks_fused": int(peaks_f),
        "troughs_fused": int(troughs_f),
        # keep raw for logic if needed later (but we won’t print numbers)
        "oppA_pct": float(100.0 * oppA),
        "oppD_pct": float(100.0 * ((np.sign(Dz) != np.sign(Vz)) & (np.abs(Dz) >= 0.8)).mean() if len(Vz) else 0.0),
        "share_v": float(share_v), "share_a": float(share_a), "share_d": float(share_d),
    }


def _neighbors_for_json(matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize DTW results into title/tag/dist for chips & UI."""
    if not matches: return []
    ds = [m["dist"] for m in matches]; mn, mx = min(ds), max(ds)
    out = []
    for m in matches:
        closeness = 1.0 if (mx - mn) < 1e-9 else 1.0 - (m["dist"] - mn) / (mx - mn + 1e-9)
        tag = "very close" if closeness >= 0.8 else "close" if closeness >= 0.6 else "nearby" if closeness >= 0.4 else "related"
        out.append({"title": m["title"], "tag": tag, "dist": round(m["dist"], 1)})
    return out
def _label_neighbors(matches: List[Dict[str, Any]]) -> List[str]:
    """Turn DTW distances into readable closeness labels for the prompt."""
    if not matches: return []
    ds = [m["dist"] for m in matches]
    mn, mx = min(ds), max(ds)
    labeled = []
    for m in matches:
        if mx - mn < 1e-9:
            closeness = 0.75
        else:
            closeness = 1.0 - (m["dist"] - mn) / (mx - mn + 1e-9)  # 1..0 best→worst
        if closeness >= 0.8:
            tag = "very close"
        elif closeness >= 0.6:
            tag = "close"
        elif closeness >= 0.4:
            tag = "nearby"
        else:
            tag = "related"
        labeled.append(f"{m['title']} — {tag}")
    return labeled


def build_insight_prompt(title: str,
                         summary: Dict[str, Any],
                         matches: List[Dict[str, Any]],
                         vad_data_block: Optional[str] = None,
                         words: str = "180–220") -> str:
    neighbor_lines = _label_neighbors(matches[:5])
    neighbors_txt = "\n".join(f"- {line}" for line in neighbor_lines) if neighbor_lines else "- (no close neighbors available)"

    data_section = f"\n\n{vad_data_block}\n" if vad_data_block else ""

    return f"""
You are a literary guide. Objective: explain the **emotional experience** (VAD = Valence/Arousal/Dominance) of this book in clear, vivid language. Do NOT summarize plot.

Write 2 short paragraphs ({words} words total), natural and evocative, no bullets, no numbers.

Focus on:
• mood/pleasantness (valence) and how it shifts,
• tension/energy (arousal) surging vs. simmering,
• sense of agency/control (dominance),
• how the feeling evolves across the whole book (start→middle→end).

Signals from analysis (guidance only; never quote numbers):
- INTENSITY: {summary['intensity']}
- TENSION: {summary['a_note']}
- AGENCY: {summary['d_note']}
In paragraph two, name the neighbors explicitly and say how this book’s **valence** and **arousal** curves feel similar or different to theirs, in one clause each.

You MUST explicitly compare to these emotionally similar classics (by plot):
{neighbors_txt}

Use the similarities to describe kinship of mood (e.g., “quiet dread like in …”, “social wit with soft melancholy like in …”), not story beats. Spoiler-free for all books.

{data_section}
Now write the insight for: **{title}**.
""".strip()

# ======== VAD COMPRESSION & PROMPT DATA HELPERS ========

from typing import Dict, Any, List, Optional
import numpy as np

def _downsample_series(y: List[float], n: int, zscore: bool = True, clip: float = 2.5) -> List[float]:
    """Downsample to n points by averaging equal segments. Z-score and clip to +/- clip."""
    if not y or n <= 0:
        return []
    arr = np.asarray(y, dtype=float)
    if zscore:
        m, s = np.nanmean(arr), np.nanstd(arr) + 1e-9
        arr = (arr - m) / s
    L = len(arr)
    if L <= n:
        out = arr
    else:
        # mean-pool into n segments
        edges = np.linspace(0, L, num=n+1, dtype=int)
        out = np.array([np.nanmean(arr[edges[i]:edges[i+1]]) for i in range(n)], dtype=float)
    if clip is not None and clip > 0:
        out = np.clip(out, -clip, clip)
    # round to 2 decimals to control tokens
    return [float(f"{v:.2f}") for v in out.tolist()]

def compress_arc_for_llm(arc: Dict[str, Any],
                         points_main: int = 60,
                         include: tuple = ("vad_fused", "v", "a", "d")) -> Dict[str, List[float]]:
    """
    Return short numeric lists (z-scored, clipped) for series requested.
    Default 60 pts for the user's book; neighbors will use fewer (see make_vad_data_block).
    """
    out = {}
    for key in include:
        if key in arc:
            out[key] = _downsample_series(arc[key], points_main)
    return out

def make_vad_data_block(title: str,
                        user_arc: Dict[str, Any],
                        neighbor_titles: List[str],
                        bench: Dict[str, Any],
                        points_main: int = 60,
                        points_neighbor: int = 36,
                        include: tuple = ("vad_fused", "v", "a", "d")) -> str:
    """
    Build a compact, token-budgeted VAD data section for the prompt.
    - User book: up to 60 points per series
    - Each neighbor: up to 36 points, and only 'vad_fused' by default if you want to shrink further
    """
    user_comp = compress_arc_for_llm(user_arc, points_main, include=include)

    blocks = []
    blocks.append(f"[BOOK] {title}")
    for k, vals in user_comp.items():
        if vals:
            blocks.append(f"{k}: {vals}")

    # neighbors: try to fetch arc by title key in bench (adjust if your bench keys differ)
    for nb in neighbor_titles:
        nb_data = bench.get(nb) or {}
        nb_arc = nb_data.get("arc") or nb_data  # support both formats
        if not (isinstance(nb_arc, dict) and "vad_fused" in nb_arc):
            continue
        nb_comp = compress_arc_for_llm(nb_arc, points_neighbor, include=("vad_fused",))
        if nb_comp.get("vad_fused"):
            blocks.append(f"[NEIGHBOR] {nb}")
            blocks.append(f"vad_fused: {nb_comp['vad_fused']}")

    return "VAD DATA (z-scored, clipped; short lists, left→right ~ start→end):\n" + "\n".join(blocks)

def neighbor_titles_from_matches(matches: List[Dict[str, Any]], k: int = 2) -> List[str]:
    """Pick top-k match titles in order."""
    if not matches:
        return []
    return [m["title"] for m in matches[:max(0, k)]]
