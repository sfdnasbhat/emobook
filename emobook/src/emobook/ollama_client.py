# src/emobook/ollama_client.py
from __future__ import annotations
import os, requests, time

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

def _ensure_model(name: str, timeout_s: int = 900) -> None:
    # 1) see what models are available
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=30)
        r.raise_for_status()
        models = [m.get("model") for m in r.json().get("models", [])]
        if name in models:
            return
    except Exception:
        # if tags fails, still try to pull
        pass
    # 2) pull if missing
    pr = requests.post(
        f"{OLLAMA_HOST}/api/pull",
        json={"name": name, "stream": False},
        timeout=timeout_s
    )
    pr.raise_for_status()

def generate(prompt: str, model: str | None = None, temperature: float = 0.3, max_tokens: int = 700) -> str:
    name = model or OLLAMA_MODEL
    # make sure the model exists (pull if needed)
    _ensure_model(name)

    body = {
        "model": name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=body, timeout=180)
        r.raise_for_status()
        return r.json().get("response", "")
    except requests.HTTPError as e:
        # if the model just got pulled but not loaded yet, retry once after a short wait
        time.sleep(2.0)
        r2 = requests.post(f"{OLLAMA_HOST}/api/generate", json=body, timeout=180)
        r2.raise_for_status()
        return r2.json().get("response", "")
