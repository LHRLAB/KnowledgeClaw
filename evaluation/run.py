#!/usr/bin/env python
"""Reproduce KnowledgeClaw evaluation on OmniKE-Bench.

Usage:
    # Evaluate on all 10 personas
    python -m evaluation.run

    # Evaluate on a single persona
    python -m evaluation.run --persona Alice_Healthcare

    # Only score existing raw outputs (skip running)
    python -m evaluation.run --score-only

Data layout (download from HuggingFace):
    data/
    └── v0.1/
        ├── unified/          # {Persona}.json — file content + metadata
        └── gold/             # {Persona}/t{3,4,5,6}.json — gold answers

Results are saved to:
    results/v0.1/raw/knowledgeclaw/{persona}/t{1..6}.json
    results/v0.1/scores.json
    results/v0.1/leaderboard.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.scorers import (
    score_t1, score_t2, score_t3, score_t4, score_t5, score_t6,
    omnike_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Data paths (relative to project root) ──
DATA_ROOT = Path("data/v0.1")
UNIFIED_ROOT = DATA_ROOT / "unified"
GOLD_ROOT = DATA_ROOT / "gold"
CORPUS_ROOT = DATA_ROOT / "corpus"

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

SUPPORTS = {"T1", "T2", "T3", "T4", "T5", "T6"}


# ── Run ─────────────────────────────────────────────────────────────────

def run_persona(persona: str, out_dir: Path):
    """Run KnowledgeClaw on a single persona."""
    from evaluation.adapter import run as adapter_run

    log.info("[KnowledgeClaw x %s] running...", persona)

    unified_path = UNIFIED_ROOT / f"{persona}.json"
    if not unified_path.exists():
        log.error("Data not found: %s", unified_path)
        log.error("Please download the OmniKE-Bench dataset and place it under data/v0.1/")
        sys.exit(1)

    unified = json.loads(unified_path.read_text())
    gold = {}
    for t in range(1, 7):
        p = GOLD_ROOT / persona / f"t{t}.json"
        if p.exists():
            gold[f"t{t}"] = json.loads(p.read_text())

    corpus_dir = str(CORPUS_ROOT / persona)
    method_out = out_dir / "raw" / "knowledgeclaw" / persona
    method_out.mkdir(parents=True, exist_ok=True)

    results = adapter_run(
        persona=persona,
        corpus_dir=corpus_dir,
        unified=unified,
        gold=gold,
        supports=SUPPORTS,
        out_dir=method_out,
    )

    # Save raw outputs
    for task_key, data in results.items():
        (method_out / f"{task_key}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    log.info("[KnowledgeClaw x %s] done, saved to %s", persona, method_out)
    return results


# ── Score ───────────────────────────────────────────────────────────────

def score_persona(persona: str, out_dir: Path) -> dict:
    """Load raw outputs and gold, compute all task scores."""
    raw_dir = out_dir / "raw" / "knowledgeclaw" / persona
    gold = {}
    for t in range(1, 7):
        p = GOLD_ROOT / persona / f"t{t}.json"
        if p.exists():
            gold[f"t{t}"] = json.loads(p.read_text())

    unified_path = UNIFIED_ROOT / f"{persona}.json"
    unified_files = []
    if unified_path.exists():
        unified_files = json.loads(unified_path.read_text()).get("files", [])

    scores = {}

    if (raw_dir / "t1.json").exists():
        pred = json.loads((raw_dir / "t1.json").read_text())
        scores.update(score_t1(pred, unified_files))

    if (raw_dir / "t2.json").exists():
        pred = json.loads((raw_dir / "t2.json").read_text())
        scores.update(score_t2(pred, unified_files))

    if (raw_dir / "t3.json").exists() and "t3" in gold:
        pred = json.loads((raw_dir / "t3.json").read_text())
        scores.update(score_t3(pred, gold["t3"]))

    if (raw_dir / "t4.json").exists() and "t4" in gold:
        pred = json.loads((raw_dir / "t4.json").read_text())
        scores.update(score_t4(pred, gold["t4"]))

    if (raw_dir / "t5.json").exists() and "t5" in gold:
        pred = json.loads((raw_dir / "t5.json").read_text())
        scores.update(score_t5(pred, gold["t5"]))

    if (raw_dir / "t6.json").exists():
        pred = json.loads((raw_dir / "t6.json").read_text())
        scores.update(score_t6(pred, unified_files))

    return scores


def generate_report(out_dir: Path, personas: list[str]):
    """Score all personas, generate leaderboard + scores.json."""
    per_persona = {}
    for persona in personas:
        per_persona[persona] = score_persona(persona, out_dir)

    # Task averages
    task_avgs = {}
    for t in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        vals = [per_persona[p].get(t) for p in personas
                if per_persona[p].get(t) is not None]
        task_avgs[t] = round(sum(vals) / len(vals), 1) if vals else None

    # Persona averages
    persona_avgs = {}
    for p in personas:
        vals = [per_persona[p].get(t) for t in ["T1", "T2", "T3", "T4", "T5", "T6"]
                if per_persona[p].get(t) is not None]
        persona_avgs[p] = round(sum(vals) / len(vals), 1) if vals else None

    oke = omnike_score(task_avgs)
    if oke:
        oke = round(oke, 1)

    all_scores = {
        "knowledgeclaw": {
            "display": "KnowledgeClaw",
            "task_avgs": task_avgs,
            "persona_avgs": persona_avgs,
            "OKE": oke,
            "per_persona": per_persona,
        }
    }

    # Write scores.json
    (out_dir / "scores.json").write_text(
        json.dumps(all_scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Scores saved to %s", out_dir / "scores.json")

    # Print Table 1 format (same as PAPER.md)
    ALL_SHORT = ["HC", "Fin", "Edu", "Law", "Psy", "Mus", "Gam", "Bld", "Sft", "Rob"]

    print()
    print("=" * 140)
    print("Table 1 — OmniKE-Bench Results")
    print("=" * 140)

    h = "| Method |"
    for t in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        h += f" {t} |"
    for p in ALL_SHORT:
        h += f" {p} |"
    h += " OKE |"
    print(h)
    print("|---|" + "---:|" * (6 + 10 + 1))

    row = "| **KnowledgeClaw** |"
    for t in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        v = task_avgs.get(t)
        row += f" {v:.1f} |" if v is not None else " — |"
    for ap in PERSONAS:
        v = persona_avgs.get(ap)
        row += f" {v:.1f} |" if v is not None else " — |"
    row += f" **{oke:.1f}** |" if oke else " — |"
    print(row)
    print("=" * 140)

    # Write leaderboard.md
    lines = [
        "# KnowledgeClaw — OmniKE-Bench Results\n",
    ]
    h = "| Method |"
    for t in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        h += f" {t} |"
    for p in ALL_SHORT:
        h += f" {p} |"
    h += " OKE |"
    lines.append(h)
    lines.append("|---|" + "---:|" * (6 + 10 + 1))

    row = "| **KnowledgeClaw** |"
    for t in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        v = task_avgs.get(t)
        row += f" {v:.1f} |" if v is not None else " — |"
    for ap in PERSONAS:
        v = persona_avgs.get(ap)
        row += f" {v:.1f} |" if v is not None else " — |"
    row += f" **{oke:.1f}** |" if oke else " — |"
    lines.append(row)

    (out_dir / "leaderboard.md").write_text("\n".join(lines), encoding="utf-8")
    log.info("Leaderboard saved to %s", out_dir / "leaderboard.md")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce KnowledgeClaw evaluation on OmniKE-Bench"
    )
    parser.add_argument(
        "--persona", default="all",
        help="persona name or 'all' (default: all)",
    )
    parser.add_argument(
        "--out", default="results/v0.1",
        help="output directory (default: results/v0.1)",
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="skip running, only score existing raw outputs",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    personas = PERSONAS if args.persona == "all" else args.persona.split(",")

    # Validate data exists
    if not UNIFIED_ROOT.exists():
        log.error("Data directory not found: %s", UNIFIED_ROOT)
        log.error(
            "Please download the OmniKE-Bench dataset from HuggingFace "
            "and place it under data/v0.1/"
        )
        sys.exit(1)

    if not args.score_only:
        for persona in personas:
            run_persona(persona, out_dir)

    generate_report(out_dir, personas)


if __name__ == "__main__":
    main()
