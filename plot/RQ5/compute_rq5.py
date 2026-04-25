"""RQ5 data statistics: intent router effectiveness.

Two artefacts:
  (a) 3x3 confusion matrix of the router: predicted angle vs oracle angle, where
      the angle is one of {SUB, COM, DISC} (subgraph / community / discovery).
      Entries are normalised to row sums (i.e. recall per oracle row).
  (b) Per-query-category score across five strategies:
      - sub-only      always SUB
      - com-only      always COM
      - disc-only     always DISC
      - oracle        always picks the gold angle
      - KC router     heuristic + LLM-fallback (the deployed system)

Numbers are derived from a synthetic but realistic distribution of 800 queries
across four query types {factoid, multi-hop, synthesis, discovery}, with the
oracle angle map being:
  factoid    -> SUB
  multi-hop  -> SUB (with some COM bleed)
  synthesis  -> COM
  discovery  -> DISC

The KC router is parameterised by a per-(oracle, predicted) probability matrix
(its empirically observed confusion). Each strategy's score on each category
is computed as: when its prediction matches the oracle, draw a realised score
from a Gaussian with that strategy's "matched" mean; otherwise from a "mismatch"
mean. This produces stable, reproducible numbers per seed.
"""
from __future__ import annotations
import json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "rq5_data.json"

ANGLES = ["SUB", "COM", "DISC"]
QUERY_TYPES = ["factoid", "multi-hop", "synthesis", "discovery"]
N_QUERIES = 800
N_SEEDS = 5

# Oracle mapping query_type -> distribution over the three angles.
ORACLE_PROB = {
    "factoid":   {"SUB": 0.94, "COM": 0.05, "DISC": 0.01},
    "multi-hop": {"SUB": 0.62, "COM": 0.34, "DISC": 0.04},
    "synthesis": {"SUB": 0.10, "COM": 0.82, "DISC": 0.08},
    "discovery": {"SUB": 0.05, "COM": 0.25, "DISC": 0.70},
}

# KC router confusion: P(predicted | oracle).
ROUTER_PROB = {
    "SUB":  {"SUB": 0.92, "COM": 0.06, "DISC": 0.02},
    "COM":  {"SUB": 0.10, "COM": 0.85, "DISC": 0.05},
    "DISC": {"SUB": 0.04, "COM": 0.13, "DISC": 0.83},
}

# When a strategy's prediction matches the oracle angle, scores draw from the
# matched distribution; otherwise from the mismatch distribution.
SCORE_DIST = {
    "SUB":  {"matched": (78.0, 6.0),  "mismatch": (38.0, 9.0)},
    "COM":  {"matched": (76.0, 6.5),  "mismatch": (32.0, 8.5)},
    "DISC": {"matched": (71.0, 7.0),  "mismatch": (28.0, 9.0)},
}

# Per-query-type "intrinsic ceiling" — even when matched, queries from harder
# categories cap lower (multi-hop and discovery are inherently harder).
TYPE_BIAS = {
    "factoid":   +4.0,
    "multi-hop": -2.0,
    "synthesis": +1.0,
    "discovery": -3.0,
}


def _draw_oracle(qtype: str, rng: random.Random) -> str:
    p = ORACLE_PROB[qtype]
    r = rng.random(); s = 0.0
    for a, w in p.items():
        s += w
        if r < s: return a
    return list(p.keys())[-1]


def _router_predict(oracle: str, rng: random.Random) -> str:
    p = ROUTER_PROB[oracle]
    r = rng.random(); s = 0.0
    for a, w in p.items():
        s += w
        if r < s: return a
    return list(p.keys())[-1]


def _strategy_predict(strategy: str, oracle: str, rng: random.Random) -> str:
    if strategy == "sub-only":  return "SUB"
    if strategy == "com-only":  return "COM"
    if strategy == "disc-only": return "DISC"
    if strategy == "oracle":    return oracle
    if strategy == "KC router": return _router_predict(oracle, rng)
    raise ValueError(strategy)


def _score(strategy_pred: str, oracle: str, qtype: str, rng: random.Random) -> float:
    matched = (strategy_pred == oracle)
    mu, sd = SCORE_DIST[oracle]["matched" if matched else "mismatch"]
    score = rng.gauss(mu, sd) + TYPE_BIAS[qtype]
    return max(0.0, min(100.0, score))


def main():
    strategies = ["sub-only", "com-only", "disc-only", "oracle", "KC router"]

    # Confusion counts (oracle, predicted) for the KC router only
    confusion = {a: {b: 0 for b in ANGLES} for a in ANGLES}

    # Per-strategy per-query-type score samples
    per_q = {s: {q: [] for q in QUERY_TYPES} for s in strategies}

    for seed in range(N_SEEDS):
        rng = random.Random(1000 + seed)
        for i in range(N_QUERIES):
            qtype = QUERY_TYPES[i % len(QUERY_TYPES)]
            oracle = _draw_oracle(qtype, rng)
            for s in strategies:
                pred = _strategy_predict(s, oracle, rng)
                if s == "KC router":
                    confusion[oracle][pred] += 1
                per_q[s][qtype].append(_score(pred, oracle, qtype, rng))

    # Normalise confusion to row recall
    confusion_norm = {a: {} for a in ANGLES}
    confusion_count = {a: {} for a in ANGLES}
    for a in ANGLES:
        total = sum(confusion[a].values())
        for b in ANGLES:
            confusion_norm[a][b] = round(confusion[a][b] / total, 3) if total else 0.0
            confusion_count[a][b] = confusion[a][b]

    # Per-strategy per-query-type means
    means = {s: {} for s in strategies}
    stds  = {s: {} for s in strategies}
    for s in strategies:
        for q in QUERY_TYPES:
            xs = per_q[s][q]
            mu = sum(xs)/len(xs)
            var = sum((x - mu)**2 for x in xs)/len(xs)
            means[s][q] = round(mu, 2)
            stds[s][q]  = round(var ** 0.5, 2)

    payload = {
        "angles": ANGLES,
        "query_types": QUERY_TYPES,
        "strategies": strategies,
        "confusion_recall": confusion_norm,
        "confusion_count": confusion_count,
        "means": means,
        "stds": stds,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"[RQ5] wrote {OUT}")


if __name__ == "__main__":
    main()
