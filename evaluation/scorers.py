"""Scorers for T1–T6.

v4 — Final evaluation scheme (zero LLM, <1s total):
  T1: reference-free (entity faith + prop faith + quantity + MCov)
  T2: reference-free (link faith + semantic + nontrivial + diversity + quantity)
  T3: token F1 (unchanged)
  T4: 0.7*F1 + 0.3*XMEv (unchanged)
  T5: CRUD keyword verify (weighted) + fact-change token F1
  T6: reference-free (known accuracy + gaps accuracy against KB)
"""
from __future__ import annotations

import re
from collections import Counter


MIN_ENTITY_LEN = 3  # Bug #1/#4/#11: ignore entities shorter than 3 chars


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _token_f1(pred: str, gold: str) -> float:
    pred_toks = _normalize(pred).split()
    gold_toks = _normalize(gold).split()
    if not gold_toks:
        return 100.0 if not pred_toks else 0.0
    if not pred_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    prec = n_common / len(pred_toks)
    rec = n_common / len(gold_toks)
    return 200.0 * prec * rec / (prec + rec)


# ---------------------------------------------------------------------------
# T1: Knowledge Acquisition (reference-free)
# ---------------------------------------------------------------------------

def score_t1(agent_result: list[dict], unified_files: list[dict]) -> dict:
    """T1 = 0.30*ent_faith + 0.30*prop_faith + 0.15*quantity + 0.25*mcov"""
    if not agent_result:
        return {"T1": 0.0, "entity_faith": 0.0, "prop_faith": 0.0, "MCov": 0.0}

    fmap = {f["id"]: _normalize(f.get("content", "")) for f in unified_files}

    ent_faiths = []
    prop_faiths = []
    quantities = []
    mods_with_output = set()

    for item in agent_result:
        fid = item.get("file_id", "")
        src = fmap.get(fid, "")
        ents = [str(e) for e in item.get("entities", [])
               if isinstance(e, str) and len(e.strip()) >= MIN_ENTITY_LEN]
        props = [str(p) for p in item.get("propositions", [])
                if isinstance(p, str) and len(p.strip()) > 10]

        if not src:
            continue

        # Entity faithfulness: fraction found in source
        if ents:
            found = sum(1 for e in ents if _normalize(e) in src)
            ent_faiths.append(found / len(ents) * 100)
            mods_with_output.add(item.get("modality", "text"))

        # Proposition faithfulness: fraction with token overlap > 20%
        if props:
            good = sum(1 for p in props if _token_f1(p, src[:5000]) > 20)
            prop_faiths.append(good / len(props) * 100)

        # Quantity
        quantities.append(min(len(ents), 15) / 15 * 100)

    ent_faith = sum(ent_faiths) / max(len(ent_faiths), 1)
    prop_faith = sum(prop_faiths) / max(len(prop_faiths), 1)
    quantity = sum(quantities) / max(len(quantities), 1)
    mcov = len(mods_with_output) / 5 * 100

    t1 = 0.30 * ent_faith + 0.30 * prop_faith + 0.15 * quantity + 0.25 * mcov
    return {"T1": t1, "entity_faith": ent_faith, "prop_faith": prop_faith, "MCov": mcov}


# ---------------------------------------------------------------------------
# T2: Cross-modal Coreference (reference-free)
# ---------------------------------------------------------------------------

def score_t2(agent_result: list[dict], unified_files: list[dict]) -> dict:
    """T2 = 0.30*faith + 0.25*semantic + 0.15*nontrivial + 0.15*diversity + 0.15*quantity"""
    if not agent_result:
        return {"T2": 0.0, "XMR": 0.0}

    # Build per-modality content index
    mod_content = {}
    for f in unified_files:
        mod = f.get("modality", "text")
        content = _normalize(f.get("content", ""))
        if content:
            mod_content.setdefault(mod, "")
            mod_content[mod] += " " + content

    n_total = len(agent_result)
    n_faithful = 0
    n_semantic = 0
    n_nontrivial = 0
    mod_pairs_seen = set()

    for link in agent_result:
        a = _normalize(str(link.get("entity_a", "")))
        b = _normalize(str(link.get("entity_b", "")))
        # Bug #4: skip trivially short entities
        if len(a) < MIN_ENTITY_LEN or len(b) < MIN_ENTITY_LEN:
            continue
        mp = link.get("modality_pair", [])

        if not a or not b:
            continue

        # Faithfulness: both entities in their modality's content
        if len(mp) == 2:
            mod_pairs_seen.add(tuple(sorted(mp)))
            a_in_src = a in mod_content.get(mp[0], "")
            b_in_src = b in mod_content.get(mp[1], "")
            if a_in_src and b_in_src:
                n_faithful += 1

        # Semantic relatedness: shared tokens
        a_toks = set(a.split())
        b_toks = set(b.split())
        if a_toks & b_toks:
            n_semantic += 1

        # Non-triviality
        if a != b:
            n_nontrivial += 1

    n = max(n_total, 1)
    faithfulness = n_faithful / n * 100
    semantic = n_semantic / n * 100
    nontrivial = n_nontrivial / n * 100
    diversity = len(mod_pairs_seen) / 10 * 100  # C(5,2) = 10
    quantity = min(n_total, 30) / 30 * 100

    t2 = 0.30 * faithfulness + 0.25 * semantic + 0.15 * nontrivial + 0.15 * diversity + 0.15 * quantity
    return {"T2": t2, "XMR": t2, "nontrivial": nontrivial, "diversity": diversity}


# ---------------------------------------------------------------------------
# T3: Multi-hop Retrieval (token F1)
# ---------------------------------------------------------------------------

def score_t3(agent_results: list[dict], gold: list[dict]) -> dict:
    if not gold:
        return {"T3": 0.0, "F1": 0.0, "EM": 0.0}
    # Bug #7: pad pred to match gold length (missing preds score 0)
    while len(agent_results) < len(gold):
        agent_results.append({"answer": ""})
    f1s, ems = [], []
    for g, p in zip(gold, agent_results):
        f1 = _token_f1(p.get("answer", ""), g.get("answer", ""))
        em = 100.0 if _normalize(p.get("answer", "")) == _normalize(g.get("answer", "")) else 0.0
        f1s.append(f1); ems.append(em)
    return {"T3": sum(f1s) / len(f1s), "T3_F1": sum(f1s) / len(f1s), "T3_EM": sum(ems) / len(ems)}


# ---------------------------------------------------------------------------
# T4: Cross-modal Reasoning (0.7*F1 + 0.3*XMEv)
# ---------------------------------------------------------------------------

def score_t4(agent_results: list[dict], gold: list[dict]) -> dict:
    if not gold:
        return {"T4": 0.0, "F1": 0.0, "XMEv": 0.0}
    # Bug #7: pad pred to match gold length
    while len(agent_results) < len(gold):
        agent_results.append({"answer": "", "modalities_used": []})
    f1s, xmev_count = [], 0
    for g, p in zip(gold, agent_results):
        f1s.append(_token_f1(p.get("answer", ""), g.get("answer", "")))
        if len(set(p.get("modalities_used", []))) >= 2:
            xmev_count += 1
    avg_f1 = sum(f1s) / len(f1s)
    xmev = xmev_count / len(gold) * 100
    return {"T4": 0.7 * avg_f1 + 0.3 * xmev, "T4_F1": avg_f1, "T4_XMEv": xmev}


# ---------------------------------------------------------------------------
# T5: Lifecycle Maintenance (CRUD keyword verify + fact-change F1)
# ---------------------------------------------------------------------------

def score_t5(agent_results: dict, gold: dict) -> dict:
    """T5 = 0.5 * crud_score + 0.5 * refresh_score"""
    if not gold:
        return {"T5": 0.0, "crud_score": 0.0, "refresh_score": 0.0}

    # Part A: CRUD probes
    crud_probes = gold.get("crud_probes", [])
    crud_answers = agent_results.get("crud_answers", [])
    # Bug #19: pad to match gold length (missing answers score 0)
    while len(crud_answers) < len(crud_probes):
        crud_answers.append({"answer": ""})

    weight_map = {"delete": 3, "update": 2, "add": 1, "query": 1}
    total_weight = 0
    weighted_score = 0

    for probe, answer in zip(crud_probes, crud_answers):
        op_type = probe.get("type", "query")
        weight = weight_map.get(op_type, 1)
        ans_norm = _normalize(str(answer.get("answer", "")))

        must_contain = probe.get("must_contain", [])
        must_not_contain = probe.get("must_not_contain", [])

        # Bug #9: empty keyword lists → skip this probe (don't give free points)
        if not must_contain and not must_not_contain:
            continue

        contain_ok = sum(1 for kw in must_contain if _normalize(kw) in ans_norm) / len(must_contain) if must_contain else 0.5
        not_contain_ok = sum(1 for kw in must_not_contain if _normalize(kw) not in ans_norm) / len(must_not_contain) if must_not_contain else 0.5

        probe_score = (contain_ok + not_contain_ok) / 2 * 100
        weighted_score += weight * probe_score
        total_weight += weight

    crud_score = weighted_score / max(total_weight, 1)

    # Part B: Fact-change refresh
    refresh_probes = gold.get("refresh_probes", [])
    refresh_answers = agent_results.get("refresh_answers", [])
    # Bug #19: pad to match gold length
    while len(refresh_answers) < len(refresh_probes):
        refresh_answers.append({"updated_answer": ""})

    refresh_f1s = []
    for probe, answer in zip(refresh_probes, refresh_answers):
        f1 = _token_f1(answer.get("updated_answer", ""), probe.get("updated_answer", ""))
        refresh_f1s.append(f1)

    refresh_score = sum(refresh_f1s) / max(len(refresh_f1s), 1) if refresh_f1s else 0.0

    t5 = 0.5 * crud_score + 0.5 * refresh_score
    return {"T5": t5, "crud_score": crud_score, "refresh_score": refresh_score}


# ---------------------------------------------------------------------------
# T6: Knowledge Discovery (reference-free, verify against KB)
# ---------------------------------------------------------------------------

def _bigram_overlap(phrase: str, corpus: str) -> bool:
    """Check if any bigram (2-word sequence) from `phrase` appears in `corpus`.

    More discriminative than single-word overlap (avoids domain vocabulary floor
    effect) but more tolerant than full-phrase substring match.
    """
    words = [w for w in phrase.split() if len(w) >= MIN_ENTITY_LEN]
    if len(words) <= 1:
        # Single word: fall back to direct substring check
        return phrase in corpus
    # Check all bigrams
    for i in range(len(words) - 1):
        bigram = f"{words[i]} {words[i+1]}"
        if bigram in corpus:
            return True
    return False


def score_t6(agent_results: list[dict], unified_files: list[dict]) -> dict:
    """T6 = (known_accuracy + gaps_accuracy) / 2

    Bug #18 fix: Use bigram overlap instead of exact substring match.
    A "known" item is correct if any of its bigrams appear in KB content.
    A "gaps" item is correct if none of its bigrams appear in KB content.
    """
    if not agent_results:
        return {"T6": 0.0, "known_acc": 0.0, "gaps_acc": 0.0}

    # Build full KB content index
    kb_content = " ".join(_normalize(f.get("content", "")) for f in unified_files)

    known_correct = 0
    known_total = 0
    gaps_correct = 0
    gaps_total = 0

    for result in agent_results:
        for entity in result.get("known", []):
            e = _normalize(str(entity))
            if len(e) < MIN_ENTITY_LEN:
                continue
            known_total += 1
            if _bigram_overlap(e, kb_content):
                known_correct += 1

        for entity in result.get("gaps", []):
            e = _normalize(str(entity))
            if len(e) < MIN_ENTITY_LEN:
                continue
            gaps_total += 1
            if not _bigram_overlap(e, kb_content):
                gaps_correct += 1

    known_acc = known_correct / max(known_total, 1) * 100
    gaps_acc = gaps_correct / max(gaps_total, 1) * 100

    t6 = (known_acc + gaps_acc) / 2
    return {"T6": t6, "known_acc": known_acc, "gaps_acc": gaps_acc}


# ---------------------------------------------------------------------------
# OmniKE Score
# ---------------------------------------------------------------------------

def omnike_score(task_scores: dict[str, float]) -> float | None:
    vals = []
    for t in ["T1", "T2", "T3", "T4", "T5", "T6"]:
        v = task_scores.get(t)
        if v is None or v <= 0:
            return None
        vals.append(v)
    product = 1.0
    for v in vals:
        product *= v
    return product ** (1.0 / len(vals))
