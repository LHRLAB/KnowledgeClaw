# KnowledgeClaw

---

## Install

```bash
conda create -n knowledgeclaw python=3.13
conda activate knowledgeclaw
pip install -r requirements.txt
```

For audio/video ingestion you also need a system `ffmpeg`.

---

## Configure

Drop your apiyi (or any OpenAI-compatible) API key into the project root:

```bash
echo "your_api_key_here" > api_key_apiyi.txt
```

Override the defaults via `.env` if needed (see [.env.example](.env.example)):

```env
LLM_MODEL=gemini-3.1-flash-lite-preview      # Agent + extractor
EMBED_MODEL=gemini-embedding-2-preview        # Vector embeddings
OMNI_MODEL=gemini-3.1-flash-lite-preview      # Image/audio/video transcribe & caption
```

All models go through the apiyi proxy — no local GPU required.

---

## Evaluation

Download the OmniKE-Bench dataset from HuggingFace and place it under `data/`:

```bash
# Option 1: git clone (requires git-lfs)
git lfs install
git clone https://huggingface.co/datasets/KnowledgeClaw/OmniKE-Bench data

# Option 2: huggingface-cli
pip install huggingface_hub
huggingface-cli download KnowledgeClaw/OmniKE-Bench --repo-type dataset --local-dir data
```

After download, the directory should look like:

```
data/
└── v0.1/
    ├── unified/          # Per-persona JSON (files + content)
    │   ├── Alice_Healthcare.json
    │   ├── Bob_Finance.json
    │   └── ...
    └── gold/             # Gold annotations for T3–T6
        ├── Alice_Healthcare/
        │   ├── t3.json
        │   ├── t4.json
        │   ├── t5.json
        │   └── t6.json
        └── ...
```

Reproduce the KnowledgeClaw results on OmniKE-Bench (6 tasks, 10 personas):

```bash
# Run on all 10 personas (takes ~30 min)
python -m evaluation.run

# Run on a single persona
python -m evaluation.run --persona Alice_Healthcare

# Re-score existing outputs without re-running
python -m evaluation.run --score-only
```

Results are saved to `results/v0.1/`:

| File | Description |
|---|---|
| `raw/knowledgeclaw/{persona}/t{1..6}.json` | Per-task raw predictions |
| `scores.json` | Aggregated scores (per-task, per-persona, OKE) |
| `leaderboard.md` | Formatted results table |

---

