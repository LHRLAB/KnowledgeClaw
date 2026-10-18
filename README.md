# KnowledgeClaw

Agentic knowledge engineering over an omnimodal typed hypergraph.

```
knowledgeclaw/
  perceive.py, extract.py, hypergraph.py   C1  omnimodal typed hypergraph
  lifecycle.py                             C2  closed-loop lifecycle (Add / Del / Refresh)
  retrieve.py, community.py, router.py     C3  multi-granular retrieval with intent routing
  agent.py                                 the agent
evaluation/                                OmniKE-Bench runner and scorer
```

## Setup

```bash
pip install -r requirements.txt     # Python >= 3.10; ffmpeg on PATH for audio/video
export MODEL_API_KEY=...            # any OpenAI-compatible endpoint, default https://openrouter.ai/api/v1
```

Place OmniKE-Bench under `data/` (or set `DATA_DIR`):

```
data/
  corpus/<persona>/{text,pdf,image,audio,video,deltas}/
  release/{CORPUS.json,questions,gold}/
  bridge/
```

## Run

```bash
python -m evaluation.run                                  # KnowledgeClaw, 10 personas, T1-T6
python -m evaluation.run --method knowledgeclaw_wo_c1     # ablations: _wo_c1, _wo_c2, _wo_c3
python -m evaluation.run --score-only                     # print the table from saved predictions
```

Backbone: Gemini 2.5 Flash Lite, encoder: Gemini Embedding 2. Predictions, artefacts and retrieval traces are written to `results/`.

## Use as a library

```python
from pathlib import Path
from knowledgeclaw import KnowledgeClaw, OpenAIBackend, Source

kc = KnowledgeClaw(OpenAIBackend(api_key, base_url, "google/gemini-2.5-flash-lite", "google/gemini-embedding-2-preview"))
kc.construct([Source("a.pdf", "pdf", text), Source("b.png", "image", path=Path("b.png"))])
kc.commit([Source("a_v2.pdf", "pdf", text2, revises="a.pdf")])   # Refresh
kc.delete("b.png")                                              # Del
evidence = kc.retrieve("Who are the authors of a_v2.pdf?")["text"]
```
