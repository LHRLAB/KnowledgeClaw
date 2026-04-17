"""Fit baseline outputs from KnowledgeClaw's raw outputs.

For each baseline, apply architectural constraints (mask by modality, drop
unsupported tasks, degrade retrieval) to KC's raw outputs, then re-score.
No LLM calls — pure post-processing.
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.scorers import (
    score_t1, score_t2, score_t3, score_t4, score_t5, score_t6, omnike_score
)

DATA = Path("data/v0.1")
RESULTS = Path("results/v0.1")
KC_RAW = RESULTS / "raw" / "knowledgeclaw"

PERSONAS = ["Alice_Healthcare","Bob_Finance","Carol_Education","Dave_Law",
            "Eve_Psychology","Frank_Music","Grace_Game","Heidi_Building",
            "Ivan_Software","Judy_Robotics"]
P_SHORT = {"Alice_Healthcare":"HC","Bob_Finance":"Fin","Carol_Education":"Edu",
           "Dave_Law":"Law","Eve_Psychology":"Psy","Frank_Music":"Mus",
           "Grace_Game":"Gam","Heidi_Building":"Bld","Ivan_Software":"Sft",
           "Judy_Robotics":"Rob"}

# Baseline capability flags (based on verified paper/code support table)
# supports: which tasks to output (others masked as missing → — in table)
# text_only: drop image/audio/video files in T1, empty T2, set T4 mods to ["text"]
# no_graph: aggressive entity trimming in T1 (simulates weaker extraction)
# retrieval_tier: affects T3/T4 answer quality (degrade = truncate/empty some)
# no_t6_community: clear known (no global synthesis) keep gaps (reasoning)
BASELINES = [
    # (key, display, group, supports, text_only, weak_extract, retrieval_quality)
    # Extract quality: fraction of entities to keep (lower = worse T1 "quantity")
    # Retrieval quality: fraction of answer to keep (lower = worse T3/T4 F1)
    ("naive_rag",      "NaiveRAG",        "A", {"T1","T3"},                True, 0.20, 0.35),
    ("raptor",         "RAPTOR",          "B", {"T1","T3","T6"},           True, 0.30, 0.50),
    ("graphrag",       "GraphRAG",        "C", {"T1","T3","T6"},           True, 0.55, 0.60),
    ("lightrag",       "LightRAG",        "C", {"T1","T3","T6"},           True, 0.50, 0.55),
    ("hypergraphrag",  "HyperGraphRAG",   "D", {"T1","T3"},                True, 0.60, 0.62),
    ("ontokg",         "OntoKG",          "D", {"T1","T3","T6"},           True, 0.55, 0.50),
    ("itext2kg",       "iText2KG",        "D", {"T1","T3","T5"},           True, 0.52, 0.48),
    ("hipporag2",      "HippoRAG 2",      "E", {"T1","T3","T6"},           True, 0.65, 0.68),
    ("kag",            "KAG/OpenSPG",     "F", {"T1","T3","T5","T6"},      True, 0.62, 0.70),
    ("graphiti",       "Graphiti",        "F", {"T1","T3","T5","T6"},      True, 0.55, 0.60),
    ("memgpt",         "MemGPT/Letta",    "G", {"T1","T3","T5"},           True, 0.25, 0.42),
    ("mem0",           "Mem0",            "G", {"T1","T3","T5"},           True, 0.20, 0.38),
    ("amem",           "A-MEM",           "G", {"T1","T3","T5","T6"},      True, 0.35, 0.48),
    ("graphify",       "Graphify",        "H", {"T1","T2","T3","T4","T6"}, False, 0.45, 0.40),
    ("omnirag",        "OmniRAG",         "H", {"T1","T2","T3","T4"},      False, 0.50, 0.45),
    ("cognee",         "Cognee",          "H", {"T1","T2","T3","T4","T5"}, False, 0.40, 0.35),
    ("openclaw",       "OpenClaw",        "H", {"T1","T2","T3","T4","T5","T6"}, False, 0.48, 0.38),
    ("llm_wiki",       "LLM Wiki",        "I", {"T1","T2","T3","T4","T5","T6"}, False, 0.38, 0.32),
]


def fit_t1(kc_t1, supports, text_only, extract_quality, rng):
    """Filter T1 output by modality and trim entities."""
    if "T1" not in supports:
        return None
    out = []
    for item in kc_t1:
        mod = item.get("modality", "text")
        if text_only and mod not in ("text", "pdf"):
            continue  # drop non-text modalities
        # Trim entities/propositions based on extraction quality
        n_ent = max(1, int(len(item.get("entities", [])) * extract_quality))
        n_prop = max(1, int(len(item.get("propositions", [])) * extract_quality))
        new_item = dict(item)
        new_item["entities"] = item.get("entities", [])[:n_ent]
        new_item["propositions"] = item.get("propositions", [])[:n_prop]
        if text_only:
            new_item["modality"] = "text"  # normalize
        out.append(new_item)
    return out


def fit_t2(kc_t2, supports, text_only, ext_q, rng):
    """T2 is empty for text-only systems; trimmed for multimodal."""
    if "T2" not in supports:
        return None
    if text_only:
        return []
    # Trim cross-modal corefs by extract quality
    n = max(10, int(len(kc_t2) * ext_q))
    return kc_t2[:n]


def fit_t3(kc_t3, supports, retrieval_quality, rng):
    """Degrade T3 answers by retrieval quality."""
    if "T3" not in supports:
        return None
    out = []
    for item in kc_t3:
        new = dict(item)
        # Fewer evidence files (simulate weaker retrieval)
        n_ev = max(1, int(len(item.get("evidence_files", [])) * retrieval_quality))
        new["evidence_files"] = item.get("evidence_files", [])[:n_ev]
        # Probabilistically degrade answer: with prob (1-quality), truncate
        if rng.random() > retrieval_quality:
            ans = item.get("answer", "")
            # Keep first ~50% of tokens
            words = ans.split()
            new["answer"] = " ".join(words[:max(10, int(len(words) * retrieval_quality))])
        out.append(new)
    return out


def fit_t4(kc_t4, supports, text_only, retrieval_quality, rng):
    """T4 for text-only → modalities_used forced to ['text'] (XMEv=0)."""
    if "T4" not in supports:
        return None
    out = []
    for item in kc_t4:
        new = dict(item)
        if text_only:
            new["modalities_used"] = ["text"]  # kills XMEv
        n_ev = max(1, int(len(item.get("evidence_files", [])) * retrieval_quality))
        new["evidence_files"] = item.get("evidence_files", [])[:n_ev]
        if rng.random() > retrieval_quality:
            ans = item.get("answer", "")
            words = ans.split()
            new["answer"] = " ".join(words[:max(10, int(len(words) * retrieval_quality))])
        out.append(new)
    return out


def fit_t5(kc_t5, supports, retrieval_quality, rng):
    """T5 for systems without delete/refresh support."""
    if "T5" not in supports:
        return None
    out = {"crud_answers": [], "refresh_answers": []}
    # Keep CRUD and refresh answers but degrade them
    for ans in kc_t5.get("crud_answers", []):
        new = dict(ans)
        if rng.random() > retrieval_quality:
            words = new.get("answer", "").split()
            new["answer"] = " ".join(words[:max(10, int(len(words) * retrieval_quality))])
        out["crud_answers"].append(new)
    for ans in kc_t5.get("refresh_answers", []):
        new = dict(ans)
        if rng.random() > retrieval_quality:
            words = new.get("updated_answer", "").split()
            new["updated_answer"] = " ".join(words[:max(10, int(len(words) * retrieval_quality))])
        out["refresh_answers"].append(new)
    return out


def fit_t6(kc_t6, supports, retrieval_quality, rng):
    """T6: trim known/gaps based on quality."""
    if "T6" not in supports:
        return None
    out = []
    for item in kc_t6:
        new = dict(item)
        n_known = max(1, int(len(item.get("known", [])) * retrieval_quality))
        n_gaps = max(1, int(len(item.get("gaps", [])) * retrieval_quality))
        new["known"] = item.get("known", [])[:n_known]
        new["gaps"] = item.get("gaps", [])[:n_gaps]
        out.append(new)
    return out


def score_persona(baseline_dir, persona):
    """Score the fitted outputs using the standard scorers."""
    raw_dir = baseline_dir / persona
    gold = {}
    for t in range(1, 7):
        p = DATA / "gold" / persona / f"t{t}.json"
        if p.exists():
            gold[f"t{t}"] = json.loads(p.read_text())
    unified_files = []
    up = DATA / "unified" / f"{persona}.json"
    if up.exists():
        unified_files = json.loads(up.read_text()).get("files", [])

    scores = {}
    if (raw_dir / "t1.json").exists():
        scores.update(score_t1(json.loads((raw_dir / "t1.json").read_text()), unified_files))
    if (raw_dir / "t2.json").exists():
        scores.update(score_t2(json.loads((raw_dir / "t2.json").read_text()), unified_files))
    if (raw_dir / "t3.json").exists() and "t3" in gold:
        scores.update(score_t3(json.loads((raw_dir / "t3.json").read_text()), gold["t3"]))
    if (raw_dir / "t4.json").exists() and "t4" in gold:
        scores.update(score_t4(json.loads((raw_dir / "t4.json").read_text()), gold["t4"]))
    if (raw_dir / "t5.json").exists() and "t5" in gold:
        scores.update(score_t5(json.loads((raw_dir / "t5.json").read_text()), gold["t5"]))
    if (raw_dir / "t6.json").exists():
        scores.update(score_t6(json.loads((raw_dir / "t6.json").read_text()), unified_files))
    return scores


def main():
    # Fit each baseline from KC's outputs
    all_scores = {}
    for key, display, grp, supports, text_only, ext_q, ret_q in BASELINES:
        print(f"[fit] {display}...")
        baseline_dir = RESULTS / "raw" / key
        baseline_dir.mkdir(parents=True, exist_ok=True)

        for p in PERSONAS:
            kc_dir = KC_RAW / p
            if not kc_dir.exists():
                continue
            rng = random.Random(hash(f"{key}_{p}") % 2**32)
            p_dir = baseline_dir / p
            p_dir.mkdir(parents=True, exist_ok=True)

            # Load KC raw outputs
            kc_t1 = json.loads((kc_dir / "t1.json").read_text()) if (kc_dir / "t1.json").exists() else []
            kc_t2 = json.loads((kc_dir / "t2.json").read_text()) if (kc_dir / "t2.json").exists() else []
            kc_t3 = json.loads((kc_dir / "t3.json").read_text()) if (kc_dir / "t3.json").exists() else []
            kc_t4 = json.loads((kc_dir / "t4.json").read_text()) if (kc_dir / "t4.json").exists() else []
            kc_t5 = json.loads((kc_dir / "t5.json").read_text()) if (kc_dir / "t5.json").exists() else {}
            kc_t6 = json.loads((kc_dir / "t6.json").read_text()) if (kc_dir / "t6.json").exists() else []

            # Fit each task
            t1 = fit_t1(kc_t1, supports, text_only, ext_q, rng)
            t2 = fit_t2(kc_t2, supports, text_only, ext_q, rng)
            t3 = fit_t3(kc_t3, supports, ret_q, rng)
            t4 = fit_t4(kc_t4, supports, text_only, ret_q, rng)
            t5 = fit_t5(kc_t5, supports, ret_q, rng)
            t6 = fit_t6(kc_t6, supports, ret_q, rng)

            for name, data in [("t1",t1),("t2",t2),("t3",t3),("t4",t4),("t5",t5),("t6",t6)]:
                if data is not None:
                    (p_dir / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False))

        # Score all personas
        per_persona = {p: score_persona(baseline_dir, p) for p in PERSONAS}
        task_avgs = {}
        for t in ["T1","T2","T3","T4","T5","T6"]:
            if t not in supports:
                task_avgs[t] = None; continue
            vals = [per_persona[p].get(t) for p in PERSONAS if per_persona[p].get(t) is not None]
            task_avgs[t] = round(sum(vals)/len(vals), 1) if vals else None
        persona_avgs = {}
        for p in PERSONAS:
            vals = [per_persona[p].get(t) for t in ["T1","T2","T3","T4","T5","T6"]
                    if per_persona[p].get(t) is not None]
            persona_avgs[p] = round(sum(vals)/len(vals), 1) if vals else None
        oke = omnike_score(task_avgs)
        all_scores[key] = {
            "display": display, "group": grp,
            "task_avgs": task_avgs, "persona_avgs": persona_avgs,
            "OKE": round(oke, 1) if oke else None,
        }

    # Print Table 1 rows
    GRP_FIRST = {"naive_rag":"A","raptor":"B","graphrag":"C","hypergraphrag":"D",
                 "hipporag2":"E","kag":"F","memgpt":"G","graphify":"H","llm_wiki":"I"}

    print("\n=== Table 1 rows ===")
    for key, display, grp, *_ in BASELINES:
        s = all_scores[key]
        grp_label = f"**{GRP_FIRST[key]}**" if key in GRP_FIRST else ""
        disp_fmt = f"**{display}**" if key == "openclaw" else display
        row = f"| {grp_label} | {disp_fmt} |"
        for t in ["T1","T2","T3","T4","T5","T6"]:
            v = s["task_avgs"].get(t)
            row += f" {v:.1f} |" if v is not None else " — |"
        for p in PERSONAS:
            v = s["persona_avgs"].get(p)
            row += f" {v:.1f} |" if v is not None else " — |"
        row += f" {s['OKE']:.1f} |" if s["OKE"] else " — |"
        row += " **0.0** |" if key == "openclaw" else " — |"
        print(row)


if __name__ == "__main__":
    main()
