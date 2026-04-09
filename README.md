# KnowledgeClaw

> 把本地文件夹变成可查询、可推演、会生长的**全模态知识超图**。
>
> Reference implementation of *KnowledgeClaw: an intelligent agentic system that
> autonomously manages a personal knowledge base across its full lifecycle via an
> Omnimodal Knowledge Hypergraph* — see [`PAPER.md`](PAPER.md).

---

## What's in the box

| Module | Paper claim | Source |
|---|---|---|
| Typed omnimodal hypergraph + cross-modal hyperedges | **C1** | [`knowledgeclaw/builder.py`](knowledgeclaw/builder.py) |
| Closed-loop agent (Construct → Query → Reason → Update) | **C2** | [`knowledgeclaw/lifecycle.py`](knowledgeclaw/lifecycle.py), [`knowledgeclaw/updater.py`](knowledgeclaw/updater.py) |
| Subgraph-level retrieval | **C3** fine tier | [`knowledgeclaw/retriever.py`](knowledgeclaw/retriever.py) |
| Community-level retrieval (Louvain + LLM reports) | **C3** coarse tier | [`knowledgeclaw/community.py`](knowledgeclaw/community.py) |
| Query intent router | **C3** dispatch | [`knowledgeclaw/router.py`](knowledgeclaw/router.py) |
| QA / Inference / Research / Eval agents | uses C3 | [`knowledgeclaw/agents.py`](knowledgeclaw/agents.py) |
| Multimodal ingestion (text/code/PDF/image/audio/video) | feeds C1 | [`knowledgeclaw/ingestor.py`](knowledgeclaw/ingestor.py), [`knowledgeclaw/modes/`](knowledgeclaw/modes/) |
| Table 1 reproducibility harness | reproduces PAPER.md | [`benchmarks/`](benchmarks/) |

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

## CLI — closed-loop lifecycle (C2)

```bash
# CONSTRUCTION — typed hypergraph from a folder
python cli.py ingest ./my_docs

# UPDATE — add a single file (incremental, runs consistency check)
python cli.py update ./new_paper.pdf

# UPDATE — delete a source (prunes chunks, hyperedges, dangling entities)
python cli.py delete old_paper.pdf

# Manual consistency sweep
python cli.py consistency

# Rebuild community summaries (C3 coarse tier) after major changes
python cli.py communities
python cli.py communities --list   # show existing summaries
```

## CLI — querying (C3)

```bash
# Hierarchical Q&A — intent router decides between subgraph/community/hybrid
python cli.py query "What did I read about contrastive learning?" --sources

# ReACT-style multi-round reasoning / 推演
python cli.py infer "Future of multimodal RAG" --verbose

# KB gap analysis
python cli.py research "quantum computing"

# KB health
python cli.py eval
python cli.py stats

# Audit log of every lifecycle event
python cli.py history
```

All commands accept `--kb /path/to/kb` (default `./kb`).

---

## Web UI

```bash
./run.sh
# http://localhost:8000
```

Endpoints (FastAPI, see [routers/kb.py](routers/kb.py)):

| Method | Path | Stage |
|---|---|---|
| `POST` | `/api/ingest` | construction |
| `POST` | `/api/update` | update |
| `POST` | `/api/delete` | update |
| `POST` | `/api/consistency` | update |
| `POST` | `/api/communities/build` | community rebuild |
| `GET`  | `/api/communities` | list community reports |
| `POST` | `/api/query` | query (router-dispatched) |
| `POST` | `/api/infer` | reasoning |
| `POST` | `/api/research` | research |
| `POST` | `/api/lifecycle` | batch lifecycle |
| `GET`  | `/api/lifecycle/history` | audit log |
| `GET`  | `/api/graph` | graph data for visualisation |
| `GET`  | `/api/stats` | KB statistics |
| `POST` | `/api/reset` | wipe storage |

---

## Reproducing PAPER.md Table 1

```bash
# 1. Generate the three synthetic benchmark splits
python cli.py bench-generate --size 50

# 2. Run the full sweep — markdown + JSON dropped into ./results/
python cli.py bench --out ./results

# Or directly via the harness
python -m benchmarks.run_table1 --out ./results
```

Latest harness output (Apr 2026, n=50 per split, deterministic, **no LLM calls**):

```
| Method                | PText 1-hop | 2-hop | 3-hop | PMM Text | Cross-modal | Synth | PDyn Cons | Ret.Acc | Avg  |
|-----------------------|-------------|-------|-------|----------|-------------|-------|-----------|---------|------|
| NaiveRAG              | 19.3        | 35.7  | 50.0  | 30.3     |  —          |  —    | 70.0      | 100.0   | 38.2 |
| GraphRAG-Edge         | 48.6        | 49.4  | 50.0  | 30.3     |  —          |  —    | 70.0      | 100.0   | 43.5 |
| GraphRAG-Community    | 27.8        | 49.4  | 46.7  | 30.3     |  —          |  —    | 70.0      | 100.0   | 40.5 |
| HyperGraphRAG         | 32.3        | 51.8  | 51.8  | 31.6     |  —          |  —    | 70.0      | 100.0   | 42.2 |
| MemGPT                | 19.3        | 35.7  | 50.0  | 20.3     | 31.0        | 81.0  | 70.0      | 100.0   | 50.9 |
| Mem0                  | 19.3        | 35.7  | 50.0  | 20.3     | 31.0        | 81.0  | 70.0      | 100.0   | 50.9 |
| KarpathyKB            | 48.6        | 49.4  | 50.0  | 50.3     | 48.6        | 81.0  | 70.0      | 100.0   | 62.2 |
| KC w/o C1 (text-only) | 87.8        | 82.0  | 82.4  | 46.7     |  —          |  —    | 100.0     | 100.0   | 62.4 |
| KC w/o C2 (static)    | 87.8        | 82.0  | 82.4  | 78.1     | 67.4        | 81.0  | 70.0      | 100.0   | 81.1 |
| KC w/o C3 (subgraph)  | 87.8        | 82.0  | 82.4  | 78.1     | 67.4        | 64.7  | 100.0     | 100.0   | 82.8 |
| KnowledgeClaw (ours)  | 87.8        | 82.0  | 82.4  | 78.1     | 67.4        | 81.0  | 100.0     | 100.0   | 84.8 |
```

The expected ablation pattern (PAPER.md Section "Ablation Drop Pattern") is reproduced:

* **KC w/o C1** loses cross-modal coverage entirely (`—`).
* **KC w/o C2** drops to 70 on `PDyn Upd.Cons.` because deletes are no longer honoured.
* **KC w/o C3** drops on `PMM Synth F1` because synthesis falls back to single-hop subgraph hits.
* **KnowledgeClaw** wins on the average and on every dimension where it makes a claim.

See [`benchmarks/README.md`](benchmarks/README.md) for benchmark schema, metric definitions, and instructions for adding your own baseline.

---

## Supported file types

| Type | Extensions | How |
|---|---|---|
| Text | `.txt` `.md` `.rst` | direct read |
| Code | `.py` `.js` `.ts` `.java` `.go` … | preserved with language tag |
| Data | `.json` `.csv` | serialised / Markdown table |
| PDF  | `.pdf` | per-page text + page rendered as PNG asset |
| Image | `.jpg` `.png` `.webp` … | Gemini caption via apiyi |
| Audio | `.mp3` `.wav` `.flac` `.m4a` … | soundfile segmenter → Gemini transcript |
| Video | `.mp4` `.avi` `.mov` `.mkv` … | ffmpeg segmenter → Gemini transcript + caption |

Each non-text artifact becomes a first-class **asset node** in the typed
hypergraph, and the canonical entities mentioned across multiple modalities
of the same source are automatically connected via a **cross-modal hyperedge**
(see [`builder._add_cross_modal_hyperedges`](knowledgeclaw/builder.py)).

---

## Architecture at a glance

```
                       ┌────────────── Lifecycle Agent (C2) ──────────────┐
                       │                                                  │
   raw files ──► Ingestor ──► Builder ──► Typed Omnimodal Hypergraph (C1) │
                                              │                           │
                                              ▼                           │
            ┌──────── Subgraph retriever (RRF) ◄──┐                       │
   query ──►│                                       Router ◄── intent     │
            └──────── Community retriever ◄────────┘                      │
                              │                                           │
                              ▼                                           │
                       QA / Inference / Research / Eval agents            │
                              │                                           │
                              ▼                                           │
                       answers + audit log ◄───────────── Updater ◄───────┘
                                                          (add / delete /
                                                           consistency)
```

---

## Acknowledgements

* Multimodal ingestion pipeline borrows from [OmniGraph-Agent](https://github.com/Zehong-Ma/OmniGraph-Agent).
* ReACT inference style adapted from [MiroFish](https://github.com/666ghj/MiroFish).
* Hyperedge representation inspired by HyperGraphRAG.
* Community detection via NetworkX Louvain.
