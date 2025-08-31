# 📚 EmoBook — Emotional Arc Analysis with VAD + LLMs

EmoBook is an experimental toolkit for **emotion analysis in literature**.  
It processes books (plain `.txt` files), computes **VAD (Valence, Arousal, Dominance) trajectories**,  
compares them to benchmark classics, and uses an **LLM (via Ollama)** to generate  
a spoiler-free summary of the *emotional experience* of the text.

This project grew out of academic coursework on **Emotion Analysis** ( Summer 2025):, but it’s designed to be reusable and extended by the community and hopefully by not violating the licences of NRC VAD lexicon.

---

## Features
- Upload any `.txt` book and get:
  - **Computed emotional arc** using NRC-VAD lexicon.
  - **Comparison with classic benchmark arcs** (e.g., *Pride and Prejudice*, *Moby Dick*).
  - **LLM-based reader’s insight** that integrates VAD signals + benchmark similarities.
- **Containerized** (Docker + Compose): easy to spin up locally or in the cloud.
- **Gradio web UI**: no FastAPI endpoints to wrestle with — just a simple web app.

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/emobook.git
cd emobook
```
## Configure environment
Create a .env file and set:
OLLAMA_MODEL=llama3.1:8b
OLLAMA_HOST=http://ollama:11434

## Build & run

```bash
cd docker
docker compose up -d --build

```
This will:
Start an Ollama sidecar + pull the specified model.
Launch the Gradio UI at http://localhost:7860.

## Development

Requirements if running without Docker:

```bash
pip install -r requirements.txt
python app_gradio.py
````

## Contributing

Contributions are **highly welcome**!  
Whether it’s fixing a bug, improving the VAD compression, or adding new benchmark texts:

- **Open an Issue** → report bugs, request features, or discuss design changes.  
- **Fork & PR** → submit pull requests with clear descriptions and tests where possible.  
- **Docs & Examples** → clarifications and tutorials are very valuable too.  

If you’re a student working on emotion analysis Project,  
feel free to add your own corpora, methods, or evaluation pipelines.  

Let’s build this into a useful open resource for literary emotion analysis.

---

## 📜 License

MIT — free to use, modify, and distribute. should only be used for research and educational purposes only as the VAD lebelling mechanism is licensed .

---

##  Acknowledgements

- NRC-VAD Lexicon v2.1  
- Classic corpora used as emotional benchmarks  
- Emotion Analysis coursework (University of Bamberg, Summer 2025)

---

 *If you like this project, give it a ⭐ on GitHub and consider opening a PR.  
Every contribution — big or small — helps us all understand stories through their emotions.*
