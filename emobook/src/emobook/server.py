from __future__ import annotations
from pathlib import Path
import os, argparse
from typing import Dict, Any, List

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse, PlainTextResponse
import uvicorn
import gradio as gr
from gradio.routes import mount_gradio_app
# Core pipeline
from .pipeline import process_book_file, RAW_DIR
from .compare import load_benchmarks, compare_against_bench
from .ollama_client import generate as ollama_generate

# ---- App paths
ROOT = Path(os.environ.get("EMOBOOK_ROOT", Path(__file__).resolve().parents[2]))
BENCH_DIR = ROOT / "benchmarks"
UPLOADS_DIR = ROOT / "uploads"
BENCH_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)




app = FastAPI(title="EmoBook API", version="1.0")

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


def build_insight_prompt(title: str, summary: Dict[str, Any], matches: List[Dict[str, Any]]) -> str:
    neighbor_lines = _label_neighbors(matches[:5])
    # Reader-facing instructions; no metrics jargon; MUST name neighbors
    return f"""
You are a literary guide. The goal: give a reader a spoiler-free sense of the *emotional experience* of this book.

Write 2–3 short paragraphs (total 180–260 words), natural and vivid, no bullet lists, no numbers.
Focus on:
• overall mood (pleasantness/bittersweetness), 
• pacing/tension (surge vs. simmer),
• sense of agency/control (empowered vs. powerless),
• how the feeling evolves across the book (arc, not plot).

Use these signals (guidance only; do not quote them verbatim, do not include numbers):
- INTENSITY: {summary['intensity']}
- TENSION: {summary['a_note']}
- AGENCY: {summary['d_note']}

You MUST explicitly name emotionally similar classics from our shelf and say how they feel similar, in one clause each. Use this set exactly (don’t invent others):
{chr(10).join(f"- {line}" for line in neighbor_lines) if neighbor_lines else "- (no close neighbors available)"}

When comparing, talk about mood kinship (e.g., “quiet dread like in…” or “social wit with soft melancholy like in…”), not plot beats. No spoilers for any book.

Close with one sentence: who will enjoy this (e.g., readers wanting a steady, reflective flow vs. a high-contrast rollercoaster).

Now write the insight for: **{title}**.
""".strip()

# ----- Gradio callback + UI -----
def _gradio_insight(file_obj, topk):
    if file_obj is None:
        return ("### Please upload a .txt file.", "", "")
    # Gradio gives a temp path at file_obj.name
    src_path = Path(getattr(file_obj, "name", "upload.txt"))
    raw_bytes = src_path.read_bytes()
    dest = UPLOADS_DIR / src_path.name
    dest.write_bytes(raw_bytes)

    # pipeline → arcs
    res = process_book_file(dest)
    arc = res["arc"]
    summary = summarize_arc(arc)

    # compare to shelf
    shelf = load_benchmarks(BENCH_DIR)
    matches = compare_against_bench(arc, shelf, topk=int(topk)) if shelf else []
    neighbors_json = _neighbors_for_json(matches)

    # prompt → local Ollama
    prompt = build_insight_prompt(dest.stem, summary, matches)
    try:
        text = ollama_generate(prompt)
    except Exception:
        text = (
            "LLM unavailable right now — quick data-based summary.\n\n"
            f"- Emotional intensity: {summary['intensity']}\n"
            f"- Tension vs tone: {summary['a_note']}\n"
            f"- Agency/control: {summary['d_note']}\n"
            + (f"- Similar classics: {', '.join(n['title'] for n in neighbors_json)}\n" if neighbors_json else "")
        )

    neighbor_md = "### Similar classics\n" + "\n".join([f"- **{n['title']}** · {n['tag']}" for n in neighbors_json]) if neighbors_json else ""
    tags_md = f"**Intensity:** {summary['intensity']}  •  **Tension:** {summary['a_note']}  •  **Agency:** {summary['d_note']}"
    title_md = f"## {dest.stem}"

    return (title_md + "\n\n" + text, neighbor_md, tags_md)

def build_demo():
    with gr.Blocks(title="EmoBook — Emotional Insight", analytics_enabled=False) as demo:
        gr.Markdown("# EmoBook")
        gr.Markdown("Upload a book (.txt) → get a reader-facing emotional insight, with similar classics.")
        with gr.Row():
            file = gr.File(label="Upload .txt", file_types=[".txt"])
            topk = gr.Slider(1, 10, value=5, step=1, label="Similar classics (Top-K)")
        btn = gr.Button("Get insight")
        out_expl = gr.Markdown()
        out_neighbors = gr.Markdown()
        out_tags = gr.Markdown()
        btn.click(_gradio_insight, inputs=[file, topk], outputs=[out_expl, out_neighbors, out_tags])
    return demo

from gradio.routes import mount_gradio_app
demo = build_demo()
app = mount_gradio_app(app, demo, path="/app")

# --- Gradio JSON-Schema bool fix (prevents 500 on /app) ---
def _patch_gradio_client_bool_schema_bug():
    try:
        import gradio_client.utils as GU
        _orig = GU.json_schema_to_python_type
        def _safe(schema, defs=None):
            # Gradio sometimes passes boolean schemas; guard them.
            if isinstance(schema, bool):
                return "object" if schema else "never"
            return _orig(schema, defs)
        GU.json_schema_to_python_type = _safe
    except Exception:
        pass

_patch_gradio_client_bool_schema_bug()
# Mount Gradio at /app
demo = build_demo()
app = mount_gradio_app(app, demo, path="/app")



# ---------- routes ----------
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse(url="/app")

@app.get("/health", include_in_schema=False)
def health():
    return {"ok": True}

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return PlainTextResponse("", status_code=204)

@app.post("/insight")
async def insight(file: UploadFile = File(...), topk: int = Form(5)):
    """
    Upload a .txt, get a single LLM-written insight (string).
    Uses the precomputed benchmarks in /benchmarks to provide similarity context.
    """
    raw_path = UPLOADS_DIR / file.filename
    raw_path.write_bytes(await file.read())

    # pipeline → arcs
    res = process_book_file(raw_path)
    arc = res["arc"]
    summary = summarize_arc(arc)

    # compare to shelf
    shelf = load_benchmarks(BENCH_DIR)
    matches = compare_against_bench(arc, shelf, topk=topk) if shelf else []

    # prompt → local Ollama
    prompt = build_insight_prompt(raw_path.stem, summary, matches)
    try:
        text = ollama_generate(prompt)
    except Exception:
        text = (
            "LLM unavailable right now — quick data-based summary.\n\n"
            f"- Emotional intensity: {summary['intensity']}\n"
            f"- Tension vs tone: {summary['a_note']}\n"
            f"- Agency/control: {summary['d_note']}\n"
            + (f"- Similar classics: {', '.join(n['title'] for n in _neighbors_for_json(matches))}\n" if matches else "")
        )

    neighbors_json = _neighbors_for_json(matches)
    
    return JSONResponse({
    "title": raw_path.stem,
    "explanation": text,
    "neighbors": neighbors_json,
    "meta": {
        "intensity": summary["intensity"],
        "tension": summary["a_note"],
        "agency": summary["d_note"],
    }
})

    # return only what you asked for
    # return JSONResponse({"title": raw_path.stem, "explanation": text})

# ---------- CLI (for precomputing shelf) ----------
def cli_precompute():
    rows = []
    for p in sorted(RAW_DIR.glob("*.txt")):
        try:
            res = process_book_file(p)
            # we don't save benchmarks here; do it with scripts or an admin path if needed
            rows.append({"file": p.name, "n_chunks": res["arc"]["n_chunks"]})
            print(f"✔ {p.name}: {res['arc']['n_chunks']} chunks")
        except Exception as e:
            print(f"✖ {p.name}: {e}")
    print(f"Processed {len(rows)} files.")

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=False)
    s1 = sub.add_parser("serve", help="Run HTTP API")
    s1.add_argument("--host", default="0.0.0.0"); s1.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.cmd == "serve":
        uvicorn.run("emobook.server:app", host=args.host, port=args.port, reload=False)
    else:
        uvicorn.run("emobook.server:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    main()
