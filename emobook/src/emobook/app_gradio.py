# app_gradio.py
from __future__ import annotations
import os
import traceback
from pathlib import Path
from typing import Dict, Any, List

import gradio as gr

# Core pipeline + helpers you already have
from emobook.pipeline import RAW_DIR, process_book_file
from emobook.compare import summarize_arc, build_insight_prompt, compare_against_bench, load_benchmarks

# Ollama client
try:
    from emobook.ollama_client import generate as ollama_generate
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# Load benchmarks if available (non-fatal)
try:
    BENCH = load_benchmarks(Path(__file__).resolve().parent / "benchmarks")
except Exception:
    BENCH = {}

def run_pipeline(upload_path: str, summary_prompt: str) -> str:
    """
    Minimal flow: upload .txt -> process -> build prompt -> LLM -> return ONE string (Markdown).
    """
    try:
        if not upload_path:
            return "❌ No file provided."

        # Persist uploaded file into RAW_DIR
        src_path = Path(upload_path)
        raw_path = RAW_DIR / src_path.name
        raw_path.write_bytes(src_path.read_bytes())

        # Process (generates arc + outputs on disk; we only use arc)
        result = process_book_file(raw_path)  # -> {clean_path, chunks_path, scored_path, arc}
        arc = result["arc"]

        # Optional neighbors if benchmarks are loaded
        matches: List[Dict[str, Any]] = []
        if BENCH:
            matches = compare_against_bench(arc, BENCH, topk=5)

        # Build prompt
        arc_summary = summarize_arc(arc)
        title = Path(result["clean_path"]).stem.replace("_", " ")

        from emobook.compare import make_vad_data_block, neighbor_titles_from_matches
        nb_titles = neighbor_titles_from_matches(matches, k=2)
        vad_block = make_vad_data_block(
            title=title,
            user_arc=arc,
            neighbor_titles=nb_titles,
            bench=BENCH,
            points_main=60,        # ~60 points per series for the user
            points_neighbor=36,    # ~36 points for each neighbor (fused only)
            include=("vad_fused","v","a","d")  # include all four for the user
        )
        
        if summary_prompt and summary_prompt.strip():
            # If user gave a custom prompt, append signals + neighbors + VAD data block
            prompt = summary_prompt.strip() + "\n\n" + vad_block + "\n"
            prompt += (
                "Use the data above as guidance for the arc shape; do not quote numbers; "
                "compare emotionally with the listed neighbors; spoiler-free; "
                "write 2 short paragraphs (180–220 words).\n"
                f"Book title: **{title}**."
            )
        else:
            prompt = build_insight_prompt(title, arc_summary, matches, vad_data_block=vad_block, words="180–220")



        # LLM call (with a small head sample for tone; not required but helpful)
        if not OLLAMA_AVAILABLE:
            return (
                "⚠️ Ollama not available. Enable OLLAMA_HOST or start the sidecar.\n\n"
                f"Signals: {arc_summary['intensity']}; {arc_summary['a_note']}; {arc_summary['d_note']}."
            )

        txt_head = Path(result["clean_path"]).read_text(encoding="utf-8", errors="ignore")[:6000]
        summary = ollama_generate(f"{prompt}\n\n---\nSample (tone only):\n{txt_head}")
        return summary

    except Exception as e:
        return f"❌ {e}\n```\n{traceback.format_exc()}\n```"

with gr.Blocks(title="EmoBook | LLM Arc Summary") as demo:
    gr.Markdown("## 📚 Upload a .txt book → get an LLM-based emotional insight")

    with gr.Row():
        # IMPORTANT: type='filepath' makes the input be a string path
        file_in = gr.File(label="Upload .txt", file_types=[".txt"], type="filepath")
        summary_prompt = gr.Textbox(
            value="Write a vivid, spoiler-free emotional insight (180–220 words).",
            label="Custom prompt (optional)",
            lines=2,
        )
    run_btn = gr.Button("Summarize ▶️", variant="primary")

    summary_out = gr.Markdown(label="LLM Summary")

    run_btn.click(
        fn=run_pipeline,
        inputs=[file_in, summary_prompt],
        outputs=[summary_out],   # EXACTLY ONE OUTPUT
    )

if __name__ == "__main__":
    # Ensure no proxy interference in Docker
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        os.environ[k] = ""

    demo.launch(server_name="0.0.0.0", server_port=7860, share=True, show_error=True)
