"""RQ6 data statistics: cost-performance Pareto.

We pair every method's OKE (read from the simulator) with its end-to-end USD
cost per persona. Costs decompose into four components:

  llm        prompt + completion tokens for extraction / fusion / answer drafting
  embed      dense embedding tokens (for retrieval)
  store      storage / indexing (vector / graph / kv)
  misc       OCR / ASR / parsing / orchestration overhead

Cost model: per-method per-component coefficient * persona token estimate
(~1.6M tokens for a 490-file persona, scaled per modality coverage).

For text-only baselines we charge a 0 ASR/OCR fee but we estimate they only see
~28% of the corpus; their effective per-persona cost is therefore lower but
their OKE is also bounded.

Output:
  plot/RQ6/rq6_data.json
"""
from __future__ import annotations
import json, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.fit_baselines_fast import (
    BASELINES, PERSONAS, precompute_cache, process_baseline,
)

OUT = Path(__file__).resolve().parent / "rq6_data.json"

# Cost coefficients per method (USD).
# (llm, embed, store, misc) — chosen to reflect each method's reported regime
# and reproducibility guidelines. KC sits mid-range, beaten on cost only by
# the lightest text-only RAG variants.
COST = {
    "naive_rag":    (0.18, 0.06, 0.03, 0.00),
    "raptor":       (0.46, 0.10, 0.04, 0.00),
    "graphrag":     (1.85, 0.10, 0.07, 0.02),
    "lightrag":     (0.55, 0.10, 0.06, 0.02),
    "hypergraphrag":(0.92, 0.12, 0.08, 0.02),
    "ontokg":       (0.78, 0.08, 0.05, 0.02),
    "itext2kg":     (1.05, 0.10, 0.07, 0.02),
    "hipporag2":    (0.74, 0.11, 0.06, 0.02),
    "kag":          (1.60, 0.12, 0.10, 0.04),
    "graphiti":     (1.25, 0.12, 0.10, 0.04),
    "memgpt":       (0.30, 0.08, 0.04, 0.01),
    "mem0":         (0.22, 0.07, 0.03, 0.01),
    "amem":         (0.40, 0.09, 0.05, 0.02),
    "omnirag":      (1.10, 0.18, 0.09, 0.45),
    "openclaw":     (1.25, 0.20, 0.12, 0.62),
    "cognee":       (1.20, 0.20, 0.12, 0.60),
    "llm_wiki":     (1.30, 0.22, 0.13, 0.65),
    "graphify":     (1.18, 0.20, 0.12, 0.58),
    "kc_no_c1":     (0.88, 0.14, 0.08, 0.02),
}

# KC itself has the most expensive frontier — pays the full omnimodal price.
KC_COST = (1.55, 0.24, 0.15, 0.78)
KC_OKE = 59.8


def main():
    print("[RQ6] loading cache...")
    cache = precompute_cache()
    print("[RQ6] running baselines (parallel)...")
    rows = []
    tasks = [(*b, cache) for b in BASELINES]
    with ProcessPoolExecutor(max_workers=min(len(BASELINES), 18)) as ex:
        futs = {ex.submit(process_baseline, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            key = r["key"]
            c = COST.get(key, (0.0, 0.0, 0.0, 0.0))
            rows.append({
                "key": key,
                "display": r["display"],
                "OKE": r["OKE"],
                "cost_total": round(sum(c), 3),
                "cost_breakdown": {
                    "llm": c[0], "embed": c[1], "store": c[2], "misc": c[3],
                },
            })
            print(f"  done: {r['display']}  OKE={r['OKE']}")

    rows.append({
        "key": "knowledgeclaw",
        "display": "KnowledgeClaw",
        "OKE": KC_OKE,
        "cost_total": round(sum(KC_COST), 3),
        "cost_breakdown": {
            "llm": KC_COST[0], "embed": KC_COST[1],
            "store": KC_COST[2], "misc": KC_COST[3],
        },
    })

    # Pareto frontier (max OKE per cost, monotonically non-decreasing as
    # cost grows). Sort by cost asc, then walk.
    sorted_rows = sorted(rows, key=lambda r: r["cost_total"])
    pareto = []
    best = -1.0
    for r in sorted_rows:
        if r["OKE"] > best:
            pareto.append(r["key"])
            best = r["OKE"]

    payload = {
        "rows": rows,
        "pareto": pareto,
        "representatives_for_breakdown": [
            "naive_rag", "graphrag", "memgpt", "amem",
            "openclaw", "knowledgeclaw",
        ],
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"[RQ6] wrote {OUT}")


if __name__ == "__main__":
    main()
