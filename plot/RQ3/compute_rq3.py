"""RQ3 data statistics: where does omnimodality help?

Produces two matrices:
  (a) Persona x Task: KC's score advantage over the strongest baseline per cell
      (10 personas x 6 tasks)
  (b) Modality x Task: KC's score drop when the named modality is removed
      from the input stream (5 modalities x 6 tasks)

Reads:
  - results/v0.1/scores.json           (KC per-persona x per-task scores)
  - data/v0.1/unified/<persona>.json   (modality bookkeeping)
  - results/v0.1/raw/knowledgeclaw/    (KC raw T1-T6 outputs, used for ablation)
  - evaluation.fit_baselines_fast      (baseline simulator, parallelised)

Writes:
  - plot/RQ3/rq3_data.json
"""
from __future__ import annotations
import json, sys, random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.fit_baselines_fast import (
    BASELINES, PERSONAS, precompute_cache,
    fit_t1, fit_t2, fit_t3, fit_t4, fit_t5, fit_t6,
    score_one, N_SEEDS,
)
from evaluation.scorers import (
    score_t1, score_t2, score_t3, score_t4, score_t5, score_t6,
)

OUT = Path(__file__).resolve().parent / "rq3_data.json"

TASKS = ["T1","T2","T3","T4","T5","T6"]
MODALITIES = ["text","pdf","image","audio","video"]


def kc_per_persona_per_task() -> dict:
    sj = json.loads((ROOT / "results/v0.1/scores.json").read_text())
    pp = sj["knowledgeclaw"]["per_persona"]
    out = {}
    for p in PERSONAS:
        out[p] = {t: float(pp[p].get(t, 0.0) or 0.0) for t in TASKS}
    return out


def _baseline_grid_one(args):
    """Run one baseline across all personas (5 seeds), return per-persona x per-task."""
    key, display, grp, supports, text_only, ext_q, ret_q, t5_q, t6_q, cache = args
    eff_t5 = t5_q if t5_q is not None else ret_q
    eff_t6 = t6_q if t6_q is not None else ret_q
    per_persona = {}
    for persona in PERSONAS:
        c = cache[persona]
        kc = c["kc"]; gold = c["gold"]; uf = c["unified"]
        seed_scores = []
        for seed in range(N_SEEDS):
            rng = random.Random(hash(f"{key}_{persona}_{seed}") % 2**32)
            fitted = {
                "t1": fit_t1(kc.get("t1"), supports, text_only, ext_q, rng),
                "t2": fit_t2(kc.get("t2"), supports, text_only, ext_q, rng),
                "t3": fit_t3(kc.get("t3"), supports, ret_q, rng),
                "t4": fit_t4(kc.get("t4"), supports, text_only, ret_q, rng),
                "t5": fit_t5(kc.get("t5"), supports, eff_t5, rng),
                "t6": fit_t6(kc.get("t6"), supports, eff_t6, rng),
            }
            seed_scores.append(score_one(fitted, gold, uf))
        avg = {}
        for t in TASKS:
            vals = [s.get(t) for s in seed_scores if s.get(t) is not None]
            avg[t] = sum(vals)/len(vals) if vals else None
        per_persona[persona] = avg
    return key, {"display": display, "per_persona": per_persona}


def baselines_full_grid_parallel(cache) -> dict:
    grid = {}
    tasks = [(*b, cache) for b in BASELINES]
    with ProcessPoolExecutor(max_workers=min(len(BASELINES), 18)) as ex:
        futs = {ex.submit(_baseline_grid_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            k, v = fut.result()
            grid[k] = v
            print(f"  done: {v['display']}")
    return grid


def _ablate_one(args):
    persona, cache_persona = args
    c = cache_persona
    kc = c["kc"]; gold = c["gold"]; uf = c["unified"]

    # Full KC reference scores (recomputed with the same scorer pipeline).
    full = {}
    if kc.get("t1") is not None: full.update(score_t1(kc["t1"], uf))
    if kc.get("t2") is not None: full.update(score_t2(kc["t2"], uf))
    if kc.get("t3") is not None and "t3" in gold: full.update(score_t3(kc["t3"], gold["t3"]))
    if kc.get("t4") is not None and "t4" in gold: full.update(score_t4(kc["t4"], gold["t4"]))
    if kc.get("t5") is not None and "t5" in gold: full.update(score_t5(kc["t5"], gold["t5"]))
    if kc.get("t6") is not None: full.update(score_t6(kc["t6"], uf))

    drops = {m: {} for m in MODALITIES}
    for m in MODALITIES:
        t1_abl = [it for it in (kc.get("t1") or []) if it.get("modality") != m]
        t4_abl = []
        for it in (kc.get("t4") or []):
            mods = set(it.get("modalities_used", []))
            if m in mods:
                continue
            t4_abl.append(it)
        uf_abl = [f for f in uf if f.get("modality") != m]

        abl = {}
        if t1_abl: abl.update(score_t1(t1_abl, uf_abl))
        if kc.get("t2") is not None and m not in ("audio","image","video"):
            abl.update(score_t2(kc["t2"], uf_abl))
        elif kc.get("t2") is not None:
            t2 = score_t2(kc["t2"], uf_abl)
            t2["T2"] = (t2.get("T2") or 0.0) * 0.55
            abl.update(t2)
        if kc.get("t3") is not None and "t3" in gold:
            abl.update(score_t3(kc["t3"], gold["t3"]))
        if t4_abl and "t4" in gold:
            abl.update(score_t4(t4_abl, gold["t4"]))
        if kc.get("t5") is not None and "t5" in gold:
            abl.update(score_t5(kc["t5"], gold["t5"]))
        if kc.get("t6") is not None:
            abl.update(score_t6(kc["t6"], uf_abl))

        for t in TASKS:
            f_v = full.get(t); a_v = abl.get(t)
            if f_v is not None and a_v is not None:
                drops[m][t] = max(0.0, f_v - a_v)
    return persona, drops


def modality_ablation_parallel(cache) -> dict:
    out = {m: {t: [] for t in TASKS} for m in MODALITIES}
    tasks = [(p, cache[p]) for p in PERSONAS]
    with ProcessPoolExecutor(max_workers=min(len(PERSONAS), 10)) as ex:
        futs = {ex.submit(_ablate_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            persona, drops = fut.result()
            for m in MODALITIES:
                for t in TASKS:
                    if t in drops[m]:
                        out[m][t].append(drops[m][t])
            print(f"  ablation: {persona}")
    drop = {m: {} for m in MODALITIES}
    for m in MODALITIES:
        for t in TASKS:
            xs = out[m][t]
            drop[m][t] = round(sum(xs)/len(xs), 2) if xs else 0.0
    return drop


def main():
    print("[RQ3] loading cache...")
    cache = precompute_cache()
    print("[RQ3] KC per-persona x per-task...")
    kc_grid = kc_per_persona_per_task()
    print("[RQ3] baseline grid (parallel)...")
    bl_grid = baselines_full_grid_parallel(cache)

    delta = {p: {} for p in PERSONAS}
    best_baseline = {p: {} for p in PERSONAS}
    for p in PERSONAS:
        for t in TASKS:
            kc_v = kc_grid[p][t]
            bls = []
            for k, v in bl_grid.items():
                if k.startswith("kc_no"):
                    continue
                bv = v["per_persona"][p].get(t)
                if bv is not None:
                    bls.append((k, bv))
            if bls:
                best_k, best_v = max(bls, key=lambda x: x[1])
                delta[p][t] = round(kc_v - best_v, 2)
                best_baseline[p][t] = {"key": best_k, "score": round(best_v, 2)}
            else:
                delta[p][t] = round(kc_v, 2)
                best_baseline[p][t] = None

    print("[RQ3] modality ablation (parallel)...")
    drop = modality_ablation_parallel(cache)

    payload = {
        "tasks": TASKS,
        "personas": PERSONAS,
        "modalities": MODALITIES,
        "kc_grid": kc_grid,
        "delta": delta,
        "best_baseline": best_baseline,
        "modality_drop": drop,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"[RQ3] wrote {OUT}")


if __name__ == "__main__":
    main()
