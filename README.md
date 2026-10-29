# KnowledgeClaw

Code for **KnowledgeClaw: Towards Agentic Knowledge Engineering via Omnimodal Hypergraphs**, evaluated on **OmniKE-Bench**.

## 1. Environment Setup

```bash
conda create -n knowledgeclaw python=3.12
conda activate knowledgeclaw

pip install -r requirements.txt
sudo apt-get install ffmpeg      # ffmpeg + ffprobe, used to read audio and video natively
```

All model calls go through an OpenAI-compatible endpoint (default: OpenRouter), with Gemini 2.5 Flash Lite (`google/gemini-2.5-flash-lite`) as the backbone and Gemini Embedding 2 (`google/gemini-embedding-2-preview`) as the encoder. Get an OpenRouter API key and export it:

```bash
export MODEL_API_KEY="your_openrouter_key_here"
# optional: export MODEL_BASE_URL=...   (default https://openrouter.ai/api/v1)
```

## 2. Data Preparation

Download OmniKE-Bench from Hugging Face into `dataset/` (or point `DATA_DIR` to another location):

```bash
pip install -U huggingface_hub
hf download LHRLAB/KnowledgeClaw --repo-type dataset --local-dir dataset
```

```text
dataset/
├── corpus/<persona>/{text,pdf,image,audio,video}/   # base corpus of each persona
├── corpus/<persona>/deltas/                         # next versions committed in T5
├── release/
│   ├── CORPUS.json                                  # evaluated files of each persona
│   ├── questions/<task>/<persona>.jsonl
│   └── gold/<task>/<persona>.jsonl
└── bridge/                                          # text projection of image / audio / video (read by evaluation/data.py)
    ├── gpt/image/<persona>/<stem>.json              # image description (field "description")
    └── mimo/{audio,video}/<persona>/
        └── <stem>.consolidated.json                 # 30 s windows (field "windows": t0, t1, text)
```

`<stem>` is the media file name without extension. A missing bridge file is not an error: the file gets an empty projection. Three games audio clips shorter than 1 s (`2dfd048ab2575a13`, `36d3b80b9caf9528`, `a55d30a1d53f2dca`) have no bridge by design. KnowledgeClaw reads image / audio / video natively from `corpus/` and falls back to the bridge only when that read fails.

## 3. Implementation

```bash
python -m evaluation.run
```

This builds KnowledgeClaw over each persona's corpus, answers T1–T6 and prints the KnowledgeClaw row of Table 1. Predictions and artefacts are written to `results/knowledgeclaw/<persona>/`; model calls are cached in `results/.cache/`, so an interrupted run resumes where it stopped.
