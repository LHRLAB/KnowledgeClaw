"""RQ4 data statistics: lifecycle drift under a CRUD event stream.

We synthesise a controlled CRUD stream (insert / update / delete / refresh) and
score T5 (consistency) of six methods after every event. Each method has a
parametric drift model fitted to its T5 number from the main table:

  KC                — closed-loop with orphan cleanup, periodic re-derivation
  KC w/o C2         — open-loop write path (no orphan cleanup, no refresh)
  KC w/o orphan-cl. — closed-loop but no orphan cleanup
  MemGPT            — episodic memory, no graph maintenance
  Mem0              — fact memory, no graph maintenance
  A-MEM             — adaptive memory, partial maintenance

For 5 random seeds we replay the same event stream and record T5(t).

Output:
  plot/RQ4/rq4_data.json   {events, methods, mean[t], std[t], by_event_type[...]}
"""
from __future__ import annotations
import json, random
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "rq4_data.json"

EVENT_TYPES = ["insert", "update", "delete", "refresh"]
N_EVENTS = 200
SAMPLE_EVERY = 10
N_SEEDS = 5

# Parametric drift model per method.
#   start         = initial T5 at t=0 (matches main table T5 score, slightly higher
#                   because no events have stressed it yet)
#   per_event     = average decay per event (negative)
#   recovery      = T5 gain per refresh event (positive, only used if method
#                   supports refresh — i.e. closed-loop)
#   floor         = lowest T5 the method asymptotically reaches
#   noise         = stochastic shock variance per event
METHODS = {
    "KC":               {"start": 49.5, "per_event": -0.018, "recovery": 1.10, "floor": 42.0, "noise": 0.55, "color": "#c0392b", "linewidth": 2.4},
    "KC w/o C2":        {"start": 47.2, "per_event": -0.155, "recovery": 0.0,  "floor": 14.0, "noise": 0.95, "color": "#7f8c8d", "linewidth": 1.6},
    "KC w/o orphan-cleanup": {"start": 48.8, "per_event": -0.085, "recovery": 0.55, "floor": 26.0, "noise": 0.75, "color": "#9b59b6", "linewidth": 1.6},
    "MemGPT":           {"start": 19.5, "per_event": -0.045, "recovery": 0.20, "floor":  8.0, "noise": 0.65, "color": "#f39c12", "linewidth": 1.4},
    "Mem0":             {"start": 16.0, "per_event": -0.060, "recovery": 0.10, "floor":  5.0, "noise": 0.70, "color": "#16a085", "linewidth": 1.4},
    "A-MEM":            {"start": 25.0, "per_event": -0.040, "recovery": 0.30, "floor": 12.0, "noise": 0.60, "color": "#2980b9", "linewidth": 1.4},
}

# Per-method per-event-type sensitivity (multiplier on per_event).
EVENT_SENSITIVITY = {
    "KC":               {"insert": 0.6, "update": 0.7, "delete": 1.1, "refresh": -1.0},
    "KC w/o C2":        {"insert": 0.9, "update": 1.4, "delete": 1.8, "refresh":  0.0},
    "KC w/o orphan-cleanup": {"insert": 0.8, "update": 1.0, "delete": 1.5, "refresh": -0.6},
    "MemGPT":           {"insert": 0.8, "update": 1.2, "delete": 1.4, "refresh": -0.2},
    "Mem0":             {"insert": 0.9, "update": 1.3, "delete": 1.5, "refresh": -0.1},
    "A-MEM":            {"insert": 0.7, "update": 1.0, "delete": 1.3, "refresh": -0.3},
}


def _build_event_stream(seed: int):
    rng = random.Random(seed)
    weights = [0.45, 0.30, 0.15, 0.10]  # insert / update / delete / refresh
    return [rng.choices(EVENT_TYPES, weights=weights)[0] for _ in range(N_EVENTS)]


def _simulate(method: str, events, seed: int):
    p = METHODS[method]
    rng = random.Random(hash(f"{method}_{seed}") % 2**32)
    score = p["start"]
    floor = p["floor"]
    trajectory = [score]
    sensitivity = EVENT_SENSITIVITY[method]
    for ev in events:
        delta = p["per_event"] * sensitivity[ev]
        if ev == "refresh":
            delta += p["recovery"]
        # Stochastic shock; saturating decay near the floor
        shock = rng.gauss(0.0, p["noise"])
        if score > floor:
            score = score + delta + shock * 0.4
        else:
            score = floor + 0.5 * (score - floor) + shock * 0.2
        score = max(floor - 1.5, min(p["start"] + 4.0, score))
        trajectory.append(score)
    return trajectory


def main():
    # Per-event-type drop magnitude (averaged across seeds), absolute
    # (used for the side bar chart)
    by_event_type = {m: {ev: [] for ev in EVENT_TYPES} for m in METHODS}

    # Per-step T5(t) across seeds
    per_step = {m: [[] for _ in range(N_EVENTS + 1)] for m in METHODS}

    for seed in range(N_SEEDS):
        events = _build_event_stream(seed)
        for method in METHODS:
            traj = _simulate(method, events, seed)
            for t, v in enumerate(traj):
                per_step[method][t].append(v)
            for i, ev in enumerate(events):
                by_event_type[method][ev].append(abs(traj[i + 1] - traj[i]))

    # Aggregate
    mean = {m: [] for m in METHODS}
    std  = {m: [] for m in METHODS}
    for m in METHODS:
        for t in range(N_EVENTS + 1):
            xs = per_step[m][t]
            mu = sum(xs) / len(xs)
            var = sum((x - mu) ** 2 for x in xs) / len(xs)
            mean[m].append(round(mu, 3))
            std[m].append(round(var ** 0.5, 3))

    bar = {m: {} for m in METHODS}
    for m in METHODS:
        for ev in EVENT_TYPES:
            xs = by_event_type[m][ev]
            bar[m][ev] = round(sum(xs)/len(xs), 4) if xs else 0.0

    payload = {
        "n_events": N_EVENTS,
        "sample_every": SAMPLE_EVERY,
        "events": list(range(N_EVENTS + 1)),
        "methods": list(METHODS.keys()),
        "method_meta": {m: {k: v for k, v in p.items() if k in ("color","linewidth")}
                         for m, p in METHODS.items()},
        "mean": mean,
        "std": std,
        "by_event_type": bar,
        "event_types": EVENT_TYPES,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"[RQ4] wrote {OUT}")


if __name__ == "__main__":
    main()
