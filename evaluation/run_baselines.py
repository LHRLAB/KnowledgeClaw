#!/usr/bin/env python
"""Run ablations and baselines on OmniKE-Bench.

Usage:
    python -m evaluation.run_baselines --method kc_no_c3 --persona Alice_Healthcare
    python -m evaluation.run_baselines --method kc_no_c3,kc_no_c2,kc_no_c1 --persona all
    python -m evaluation.run_baselines --method naive_rag --persona all
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.scorers import (
    score_t1, score_t2, score_t3, score_t4, score_t5, score_t6, omnike_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = Path("data/v0.1")
UNIFIED_ROOT = DATA_ROOT / "unified"
GOLD_ROOT = DATA_ROOT / "gold"

PERSONAS = [
    "Alice_Healthcare", "Bob_Finance", "Carol_Education", "Dave_Law",
    "Eve_Psychology", "Frank_Music", "Grace_Game", "Heidi_Building",
    "Ivan_Software", "Judy_Robotics",
]
P_SHORT = {
    "Alice_Healthcare": "HC", "Bob_Finance": "Fin", "Carol_Education": "Edu",
    "Dave_Law": "Law", "Eve_Psychology": "Psy", "Frank_Music": "Mus",
    "Grace_Game": "Gam", "Heidi_Building": "Bld", "Ivan_Software": "Sft",
    "Judy_Robotics": "Rob",
}

# Method registry: (display, supports)
METHODS = {
    "naive_rag":      ("NaiveRAG",        {"T1","T3","T4"}),
    "raptor":         ("RAPTOR",          {"T1","T3","T4","T6"}),
    "graphrag":       ("GraphRAG",        {"T1","T2","T3","T4","T6"}),
    "lightrag":       ("LightRAG",        {"T1","T2","T3","T4","T6"}),
    "hypergraphrag":  ("HyperGraphRAG",   {"T1","T2","T3","T4"}),
    "ontokg":         ("OntoKG",          {"T1","T2","T3","T4","T6"}),
    "itext2kg":       ("iText2KG",        {"T1","T2","T3","T4","T5"}),
    "hipporag2":      ("HippoRAG 2",     {"T1","T2","T3","T4","T6"}),
    "kag":            ("KAG/OpenSPG",     {"T1","T2","T3","T4","T5","T6"}),
    "graphiti":       ("Graphiti",        {"T1","T2","T3","T4","T5","T6"}),
    "memgpt":         ("MemGPT/Letta",    {"T1","T3","T4","T5"}),
    "mem0":           ("Mem0",            {"T1","T3","T4","T5"}),
    "amem":           ("A-MEM",           {"T1","T2","T3","T4","T5","T6"}),
    "graphify":       ("Graphify",        {"T1","T2","T3","T4","T6"}),
    "omnirag":        ("OmniRAG",         {"T1","T2","T3","T4","T5","T6"}),
    "cognee":         ("Cognee",          {"T1","T2","T3","T4","T5","T6"}),
    "openclaw":       ("OpenClaw",        {"T1","T2","T3","T4","T5","T6"}),
    "llm_wiki":       ("LLM Wiki",        {"T1","T2","T3","T4","T5","T6"}),
    "kc_no_c1":       ("KC w/o C1",       {"T1","T2","T3","T4","T5","T6"}),
    "kc_no_c2":       ("KC w/o C2",       {"T1","T2","T3","T4","T5","T6"}),
    "kc_no_c3":       ("KC w/o C3",       {"T1","T2","T3","T4","T5","T6"}),
    "llm_wiki":       ("LLM Wiki",        {"T1","T2","T3","T4","T5","T6"}),
}


def run_method(method_key: str, persona: str, out_dir: Path, quick: int = 0):
    display, supports = METHODS[method_key]
    log.info("[%s × %s] running...%s", display, persona, f" (quick={quick})" if quick else "")

    unified = json.loads((UNIFIED_ROOT / f"{persona}.json").read_text())
    gold = {}
    for t in range(1, 7):
        p = GOLD_ROOT / persona / f"t{t}.json"
        if p.exists():
            data = json.loads(p.read_text())
            # --quick mode: limit gold samples per task
            if quick:
                if isinstance(data, list):
                    data = data[:quick]
                elif isinstance(data, dict):
                    # T5 format: {crud_probes: [...], refresh_probes: [...]}
                    for k in ("crud_probes", "refresh_probes"):
                        if k in data:
                            data[k] = data[k][:quick]
            gold[f"t{t}"] = data

    if quick:
        log.info("[quick] gold capped at %d per task, full file set kept", quick)

    method_out = out_dir / "raw" / method_key / persona
    method_out.mkdir(parents=True, exist_ok=True)

    adapter = importlib.import_module(f"evaluation.ablations.{method_key}")
    results = adapter.run(
        persona=persona,
        corpus_dir="",
        unified=unified,
        gold=gold,
        supports=supports,
        out_dir=method_out,
    )

    for task_key, data in results.items():
        (method_out / f"{task_key}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    log.info("[%s × %s] done, saved to %s", display, persona, method_out)
    return results


def score_method_persona(method_key: str, persona: str, out_dir: Path) -> dict:
    _, supports = METHODS[method_key]
    raw_dir = out_dir / "raw" / method_key / persona
    gold = {}
    for t in range(1, 7):
        p = GOLD_ROOT / persona / f"t{t}.json"
        if p.exists():
            gold[f"t{t}"] = json.loads(p.read_text())

    unified_files = []
    unified_path = UNIFIED_ROOT / f"{persona}.json"
    if unified_path.exists():
        unified_files = json.loads(unified_path.read_text()).get("files", [])

    scores = {}
    if "T1" in supports and (raw_dir / "t1.json").exists():
        scores.update(score_t1(json.loads((raw_dir / "t1.json").read_text()), unified_files))
    if "T2" in supports and (raw_dir / "t2.json").exists():
        scores.update(score_t2(json.loads((raw_dir / "t2.json").read_text()), unified_files))
    if "T3" in supports and (raw_dir / "t3.json").exists() and "t3" in gold:
        scores.update(score_t3(json.loads((raw_dir / "t3.json").read_text()), gold["t3"]))
    if "T4" in supports and (raw_dir / "t4.json").exists() and "t4" in gold:
        scores.update(score_t4(json.loads((raw_dir / "t4.json").read_text()), gold["t4"]))
    if "T5" in supports and (raw_dir / "t5.json").exists() and "t5" in gold:
        scores.update(score_t5(json.loads((raw_dir / "t5.json").read_text()), gold["t5"]))
    if "T6" in supports and (raw_dir / "t6.json").exists():
        scores.update(score_t6(json.loads((raw_dir / "t6.json").read_text()), unified_files))
    return scores


def generate_table(out_dir: Path, methods: list[str], personas: list[str]):
    all_scores = {}
    for mk in methods:
        display, supports = METHODS[mk]
        per_persona = {p: score_method_persona(mk, p, out_dir) for p in personas}

        task_avgs = {}
        for t in ["T1","T2","T3","T4","T5","T6"]:
            if t not in supports:
                task_avgs[t] = None; continue
            vals = [per_persona[p].get(t) for p in personas if per_persona[p].get(t) is not None]
            task_avgs[t] = round(sum(vals)/len(vals), 1) if vals else None

        persona_avgs = {}
        for p in personas:
            vals = [per_persona[p].get(t) for t in ["T1","T2","T3","T4","T5","T6"]
                    if per_persona[p].get(t) is not None]
            persona_avgs[p] = round(sum(vals)/len(vals), 1) if vals else None

        oke = omnike_score(task_avgs)
        all_scores[mk] = {
            "display": display, "task_avgs": task_avgs,
            "persona_avgs": persona_avgs, "OKE": round(oke, 1) if oke else None,
        }

    ALL_SHORT = ["HC","Fin","Edu","Law","Psy","Mus","Gam","Bld","Sft","Rob"]
    print()
    print("=" * 140)
    h = "| Method |"
    for t in ["T1","T2","T3","T4","T5","T6"]: h += f" {t} |"
    for p in ALL_SHORT: h += f" {p} |"
    h += " OKE |"
    print(h)
    print("|---|" + "---:|" * (6 + 10 + 1))

    for mk in methods:
        s = all_scores[mk]
        ta, pa, oke = s["task_avgs"], s["persona_avgs"], s["OKE"]
        row = f"| {s['display']} |"
        for t in ["T1","T2","T3","T4","T5","T6"]:
            v = ta.get(t)
            row += f" {v:.1f} |" if v is not None else " — |"
        for ap in PERSONAS:
            v = pa.get(ap)
            row += f" {v:.1f} |" if v is not None else " — |"
        row += f" {oke:.1f} |" if oke else " — |"
        print(row)
    print("=" * 140)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--persona", default="all")
    p.add_argument("--out", default="results/v0.1")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--quick", type=int, default=0,
                   help="limit gold samples per task (e.g. --quick 10)")
    args = p.parse_args()

    out_dir = Path(args.out)
    methods = args.method.split(",")
    personas = PERSONAS if args.persona == "all" else args.persona.split(",")

    if not args.score_only:
        for mk in methods:
            for persona in personas:
                run_method(mk, persona, out_dir, quick=args.quick)

    generate_table(out_dir, methods, personas)


if __name__ == "__main__":
    main()
