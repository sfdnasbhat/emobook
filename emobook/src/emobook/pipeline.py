# EmoBook pipeline: Gutenberg strip → sentence split → chunk (120/60) → NRC-VAD v2.1 scoring
from __future__ import annotations
import os, re, unicodedata
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

# Default to the package directory (…/src/emobook) so resources/data resolve correctly
PKG_ROOT = Path(__file__).resolve().parent          # …/src/emobook
ROOT = Path(os.environ.get("EMOBOOK_ROOT", PKG_ROOT))

RES_DIR   = ROOT / "resources"
DATA_DIR  = ROOT / "data"
RAW_DIR   = DATA_DIR / "raw"
CLEAN_DIR = DATA_DIR / "clean"
CHUNK_DIR = DATA_DIR / "chunks"
SCORED_DIR= DATA_DIR / "scored"

# Ensure directories exist (won't error if already present)
for _d in (DATA_DIR, RAW_DIR, CLEAN_DIR, CHUNK_DIR, SCORED_DIR, RES_DIR):
    _d.mkdir(parents=True, exist_ok=True)
    
CHUNK_TARGET = int(os.environ.get("CHUNK_TARGET", 120))
CHUNK_STRIDE = int(os.environ.get("CHUNK_STRIDE", 60))
CHUNK_CAP    = int(os.environ.get("CHUNK_CAP", 220))
ROLL_FRAC    = float(os.environ.get("ROLL_FRAC", 0.01))

# ---------- Gutenberg strip ----------
MODERN_START = re.compile(r"^\s*\*\*\*\s*start of (?:this|the) project gutenberg ebook.*$", re.I | re.M)
LEGACY_START = re.compile(r"^\s*the project gutenberg e?book of .*$", re.I | re.M)
END_RES = [re.compile(p, re.I | re.M) for p in [
    r"^\s*\*\*\*\s*end of (?:this|the) project gutenberg ebook.*$",
    r"^\s*end of (?:this|the) project gutenberg ebook.*$",
    r"^\s*end of project gutenberg'?s .*?$",
    r"^\s*end of the project gutenberg ebook of .*$",
    r"^\s*project gutenberg(?:™|) license.*$",
]]
CONTENTS_HDR = re.compile(r"(?:^|\n)\s*contents\s*\n", re.I)
CHAP_CUE = re.compile(r"(?:^|\n)\s*(?:chapter|book|part|canto|volume|act|scene)\s+[ivxlcdm0-9]+\b", re.I)

def extract_story(raw_text: str) -> str:
    t = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    if t.startswith("\ufeff"): t = t.lstrip("\ufeff")
    m = MODERN_START.search(t)
    if m:
        start = t.find("\n", m.end()); start = (start + 1) if start != -1 else m.end()
    else:
        mh = LEGACY_START.search(t)
        if mh:
            start = t.find("\n", mh.end()); start = (start + 1) if start != -1 else mh.end()
            m2 = MODERN_START.search(t, pos=start)
            if m2:
                start = t.find("\n", m2.end()); start = (start + 1) if start != -1 else m2.end()
        else:
            start = 0
    ends = [m.start() for r in END_RES if (m := r.search(t, pos=start))]
    end = min(ends) if ends else len(t)
    core = t[start:end].lstrip("\n")
    mC = CONTENTS_HDR.search(core[:200_000])
    if mC:
        after = core[mC.end():]
        mQ = CHAP_CUE.search(after)
        if mQ: core = after[mQ.start():]
    core = unicodedata.normalize("NFC", core)
    core = re.sub(r"[ \t]+", " ", core)
    core = re.sub(r"\n{3,}", "\n\n", core).strip()
    return core

# ---------- Sentence splitting ----------
try:
    import pysbd
    _SEG = pysbd.Segmenter(language="en", clean=False)
except Exception:
    _SEG = None

def split_sentences(text: str) -> List[str]:
    if _SEG: return _SEG.segment(text)
    t = text.replace("...", "<ELLIP>")
    t = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", t)
    t = re.sub(r'([.!?]["\')\]]?)(\s+)(?=[A-Z])', r'\1<EOS>\2', t)
    parts = [p.strip() for p in t.split("<EOS>") if p.strip()]
    return [p.replace("<ELLIP>", "...").replace("<DOT>", ".") for p in parts]

# ---------- Chunking ----------
def chunk_by_words(sentences: List[str], target=CHUNK_TARGET, stride=CHUNK_STRIDE, cap=CHUNK_CAP) -> List[str]:
    chunks, buf, curw = [], [], 0
    for s in sentences:
        w = len(s.split())
        if curw + w <= target or not buf:
            buf.append(s); curw += w
        else:
            text = " ".join(buf); words = text.split()
            chunks.append(" ".join(words[:cap]))
            keep = " ".join(words[-stride:]) if stride > 0 else ""
            buf = ([keep] if keep else []) + [s]
            curw = len(keep.split()) + w
    if buf:
        text = " ".join(buf); chunks.append(" ".join(text.split()[:cap]))
    return chunks

# ---------- NRC-VAD v2.1 ----------
LEX_PATH = Path(os.environ.get(
    "EMOBOOK_VAD_LEXICON",
    RES_DIR / "NRC-VAD-Lexicon-v2.1" / "NRC-VAD-Lexicon-v2.1.txt"
))
TOKEN_RE = re.compile(r"[a-z]+(?:'[a-z]+)?", re.I)
NEGATORS = set("not no never none nobody nothing neither nor n't cannot can't don't won't isn't wasn't aren't weren't".split())

def load_vad_v21(path: Path=LEX_PATH):
    df = pd.read_csv(path, sep="\t")
    df["term"] = df["term"].astype(str).str.strip().str.lower()
    is_mwe = df["term"].str.contains(r"\s")
    uni = {t:(float(v),float(a),float(d))
           for t,v,a,d in df.loc[~is_mwe,["term","valence","arousal","dominance"]].itertuples(index=False)}
    def tokseq(s): return tuple(TOKEN_RE.findall(s))
    mwe = {tokseq(t):(float(v),float(a),float(d))
           for t,v,a,d in df.loc[is_mwe,["term","valence","arousal","dominance"]].itertuples(index=False)}
    max_mwe_len = max((len(k) for k in mwe.keys()), default=1)
    return uni, mwe, max_mwe_len

UNI, MWE, MAX_MWE = load_vad_v21()

def tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(unicodedata.normalize("NFC", text))]

def score_chunk_v21(text: str, handle_negation=True, window_after_neg=3) -> Dict[str, Any]:
    toks = tokenize(text)
    i, hits, n_tokens, flip = 0, [], 0, 0
    while i < len(toks):
        tok = toks[i]; n_tokens += 1
        if handle_negation and tok in NEGATORS:
            flip = window_after_neg; i += 1; continue
        matched = False
        if MAX_MWE > 1:
            L = min(MAX_MWE, len(toks)-i)
            for n in range(L, 1, -1):
                key = tuple(toks[i:i+n]); trip = MWE.get(key)
                if trip:
                    v,a,d = trip
                    if flip > 0: v = -v; flip -= 1
                    hits.append((v,a,d)); i += n; matched = True; break
        if matched: continue
        trip = UNI.get(tok)
        if trip:
            v,a,d = trip
            if flip > 0: v = -v; flip -= 1
            hits.append((v,a,d))
        else:
            if flip > 0: flip -= 1
        i += 1
    if not hits:
        return dict(v=None,a=None,d=None,n_hits=0,n_tokens=n_tokens,coverage=0.0)
    arr = np.array(hits, float)
    v,a,d = arr.mean(axis=0).tolist()
    return dict(v=float(v), a=float(a), d=float(d),
                n_hits=len(hits), n_tokens=n_tokens,
                coverage=float(len(hits)/max(1,n_tokens)))

# ---------- High-level ----------
def clean_raw_file(raw_path: Path, out_dir: Path=CLEAN_DIR) -> Path:
    text = Path(raw_path).read_text(encoding="utf-8", errors="ignore")
    core = extract_story(text)
    out = out_dir / f"{Path(raw_path).stem}.clean.txt"
    out.write_text(core, encoding="utf-8")
    return out

def chunk_clean_file(clean_path: Path, out_dir: Path=CHUNK_DIR,
                     target=CHUNK_TARGET, stride=CHUNK_STRIDE, cap=CHUNK_CAP) -> Path:
    txt = Path(clean_path).read_text(encoding="utf-8")
    sents = split_sentences(txt)
    chks = chunk_by_words(sents, target=target, stride=stride, cap=cap)
    df = pd.DataFrame({"book": Path(clean_path).stem, "chunk_id": range(len(chks)), "text": chks})
    out = out_dir / f"{Path(clean_path).stem}.chunks.csv"
    df.to_csv(out, index=False)
    return out

def score_chunks_csv(chunk_csv: Path, out_dir: Path=SCORED_DIR) -> Path:
    df = pd.read_csv(chunk_csv)
    rows = [score_chunk_v21(t, handle_negation=True) for t in df["text"].astype(str)]
    S = pd.DataFrame(rows)
    out_df = pd.concat([df, S], axis=1)
    for c in ("v","a","d"): out_df[f"{c}01"] = (out_df[c] + 1.0) / 2.0
    out = out_dir / f"{Path(chunk_csv).stem.replace('.chunks','')}.scored_v21.csv"
    out_df.to_csv(out, index=False)
    return out

def compute_arcs_from_scored(scored_csv: Path, roll_frac: float=ROLL_FRAC) -> Dict[str, Any]:
    df = pd.read_csv(scored_csv).sort_values("chunk_id").reset_index(drop=True)
    for c in ("v","a","d"): df[c] = df[c].fillna(0.0)
    n = len(df)
    win = max(5, int(round(n * roll_frac)));  win = win if win % 2 == 1 else max(1, win-1)
    roll = df[["v","a","d","coverage"]].rolling(win, center=True, min_periods=1).mean()
    roll.columns = [f"{c}_roll" for c in ["v","a","d","coverage"]]
    df = pd.concat([df, roll], axis=1)
    vad_fused = np.sign(df["v_roll"]) * np.sqrt(df["v_roll"]**2 + df["a_roll"]**2 + df["d_roll"]**2)
    return {
        "book": df["book"].iloc[0] if len(df) else Path(scored_csv).stem,
        "v": df["v_roll"].tolist(),
        "a": df["a_roll"].tolist(),
        "d": df["d_roll"].tolist(),
        "coverage": df["coverage_roll"].tolist(),
        "vad_fused": vad_fused.tolist(),
        "window": int(win),
        "n_chunks": int(n),
    }

def process_book_file(raw_path: Path) -> Dict[str, Any]:
    clean = clean_raw_file(raw_path)
    chunks = chunk_clean_file(clean)
    scored = score_chunks_csv(chunks)
    arc = compute_arcs_from_scored(scored)
    return {"clean_path": str(clean), "chunks_path": str(chunks), "scored_path": str(scored), "arc": arc}
