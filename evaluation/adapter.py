"""KnowledgeClaw adapter — runs the REAL system (C1+C2+C3).

Evaluates KnowledgeClaw by:
  1. Writing unified file content to a temp directory
  2. Ingesting via kc.ingest() → builds typed hypergraph (C1)
  3. Building communities → enables community-level retrieval (C3)
  4. Running T1-T6 tasks using the actual engine APIs

This ensures the evaluation reflects all three contributions:
  C1: Omnimodal typed hypergraph (builder + ingestor + modes)
  C2: Closed-loop agentic lifecycle (delete_source + consistency)
  C3: Hierarchical multi-granular retrieval + intent routing
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from openai import OpenAI
from evaluation.eval_config import EVAL_LLM

log = logging.getLogger(__name__)

GRAPH_FIELD_SEP = "<SEP>"


def _parse_json(raw: str):
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def _llm_call(client, prompt: str, max_tokens: int = 500) -> str:
    resp = client.chat.completions.create(
        model=EVAL_LLM, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def run(persona, corpus_dir, unified, gold, supports, out_dir):
    from knowledgeclaw.engine import KnowledgeClaw

    files = unified.get("files", [])
    results = {}

    # ── Step 0: Write unified content to temp dir for ingestion ──────
    tmp_dir = tempfile.mkdtemp(prefix=f"kc_eval_{persona}_")
    storage_dir = tempfile.mkdtemp(prefix=f"kc_storage_{persona}_")
    log.info("[KC] writing %d files to temp dir for ingestion...", len(files))

    # Map file_id → temp filename for delete_source later
    id_to_filename = {}
    for f in files:
        content = (f.get("content") or "").strip()
        if len(content) < 50:
            continue
        # Use file_id as filename, .txt extension since content is always text
        fname = f["id"] + ".txt"
        fpath = Path(tmp_dir) / fname
        fpath.write_text(content, encoding="utf-8")
        id_to_filename[f["id"]] = fname

    # ── Step 1: Ingest via KnowledgeClaw engine (C1) ─────────────────
    log.info("[KC] initializing KnowledgeClaw engine...")
    kc = KnowledgeClaw(kb_dir=tmp_dir, storage_dir=storage_dir)

    log.info("[KC] ingesting %d files (C1: omnimodal typed hypergraph)...", len(id_to_filename))
    ingest_stats = kc.ingest(tmp_dir)
    log.info("[KC] ingestion done: %s", {k: v for k, v in ingest_stats.items() if isinstance(v, int)})

    # ── Step 1.5: Build communities (C3 coarse tier) ─────────────────
    log.info("[KC] building communities (C3: coarse-tier retrieval)...")
    try:
        comm_stats = kc.build_communities()
        log.info("[KC] communities built: %d", comm_stats.get("communities", 0))
    except Exception as e:
        log.warning("[KC] community building failed (non-fatal): %s", e)

    # ── T1: Knowledge Acquisition (extract from graph) ───────────────
    if "T1" in supports:
        log.info("[KC] T1: extracting entities from hypergraph...")
        t1_out = []

        # Get all entity and hyperedge nodes from the graph
        entity_nodes = kc._graph.nodes_by_role("entity")
        hyperedge_nodes = kc._graph.nodes_by_role("hyperedge")

        # Group by source file
        file_entities = {}
        for nid in entity_nodes:
            data = kc._graph.get_node(nid)
            if not data:
                continue
            sources = (data.get("sources") or data.get("source", "")).split(GRAPH_FIELD_SEP)
            modalities = (data.get("modalities") or data.get("modality", "text")).split(GRAPH_FIELD_SEP)
            name = data.get("name", nid)
            for src in sources:
                src_base = src.replace(".txt", "")
                if src_base:
                    file_entities.setdefault(src_base, {"entities": [], "modalities": set()})
                    file_entities[src_base]["entities"].append(name)
                    file_entities[src_base]["modalities"].update(modalities)

        file_props = {}
        for nid in hyperedge_nodes:
            data = kc._graph.get_node(nid)
            if not data:
                continue
            sentence = data.get("sentence", "")
            source = (data.get("source") or "").replace(".txt", "")
            if source and sentence:
                file_props.setdefault(source, []).append(sentence)

        # Build T1 output per file
        for fid in id_to_filename:
            ents = file_entities.get(fid, {}).get("entities", [])[:15]
            mods = file_entities.get(fid, {}).get("modalities", set())
            props = file_props.get(fid, [])[:8]
            # Determine modality from unified
            modality = "text"
            for f in files:
                if f["id"] == fid:
                    modality = f.get("modality", "text")
                    break
            t1_out.append({
                "file_id": fid,
                "modality": modality,
                "entities": ents,
                "propositions": props,
            })

        log.info("[KC] T1: %d files, %d total entities, %d total propositions",
                 len(t1_out),
                 sum(len(x["entities"]) for x in t1_out),
                 sum(len(x["propositions"]) for x in t1_out))
        results["t1"] = t1_out

    # ── T2: Cross-modal Coreference (from graph) ────────────────────
    if "T2" in supports:
        log.info("[KC] T2: extracting cross-modal corefs from hypergraph...")
        t2_out = []

        # Find entities that span multiple modalities
        for nid in entity_nodes:
            data = kc._graph.get_node(nid)
            if not data:
                continue
            modalities = (data.get("modalities") or "").split(GRAPH_FIELD_SEP)
            modalities = [m.strip() for m in modalities if m.strip()]
            if len(set(modalities)) >= 2:
                name = data.get("name", nid)
                for i in range(len(modalities)):
                    for j in range(i + 1, len(modalities)):
                        t2_out.append({
                            "entity_a": name,
                            "entity_b": name,
                            "canonical": name,
                            "modality_pair": [modalities[i], modalities[j]],
                        })

        # Also find cross_modal hyperedges
        for nid in hyperedge_nodes:
            data = kc._graph.get_node(nid)
            if not data:
                continue
            if data.get("kind") == "cross_modal":
                mods = (data.get("modalities") or "").split(GRAPH_FIELD_SEP)
                mods = [m.strip() for m in mods if m.strip()]
                # Get connected entities
                neighbors = kc._graph._graph.neighbors(nid) if hasattr(kc._graph, '_graph') else []
                ent_names = []
                for neighbor in neighbors:
                    nd = kc._graph.get_node(neighbor)
                    if nd and nd.get("role") == "entity":
                        ent_names.append(nd.get("name", neighbor))
                if len(ent_names) >= 2 and len(mods) >= 2:
                    t2_out.append({
                        "entity_a": ent_names[0],
                        "entity_b": ent_names[1],
                        "canonical": ent_names[0],
                        "modality_pair": mods[:2],
                    })

        log.info("[KC] T2: %d cross-modal corefs", len(t2_out))
        results["t2"] = t2_out

    # ── T3: Multi-hop Retrieval (C3: intent-routed query) ───────────
    if "T3" in supports and "t3" in gold:
        log.info("[KC] T3: answering %d questions via intent-routed retrieval (C3)...", len(gold["t3"]))

        def _t3_one(q):
            try:
                result = kc.query(q["question"], top_k=10)
                return {
                    "answer": result.get("answer", ""),
                    "evidence_files": [s.replace(".txt", "") for s in result.get("sources", [])],
                }
            except Exception:
                return {"answer": "", "evidence_files": []}

        t3_out = [None] * len(gold["t3"])
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_t3_one, q): i for i, q in enumerate(gold["t3"])}
            done = 0
            for fut in as_completed(futs):
                t3_out[futs[fut]] = fut.result()
                done += 1
                if done % 50 == 0:
                    log.info("[KC] T3: %d/%d", done, len(gold["t3"]))
        results["t3"] = t3_out

    # ── T4: Cross-modal Reasoning (C3: intent-routed query) ─────────
    if "T4" in supports and "t4" in gold:
        log.info("[KC] T4: answering %d cross-modal questions (C3)...", len(gold["t4"]))

        def _t4_one(q):
            try:
                result = kc.query(q["question"], top_k=10)
                sources = result.get("sources", [])
                # Determine modalities from source filenames
                mods_used = set()
                for src in sources:
                    src_clean = src.replace(".txt", "")
                    for f in files:
                        if f["id"] == src_clean:
                            mods_used.add(f.get("modality", "text"))
                            break
                return {
                    "answer": result.get("answer", ""),
                    "evidence_files": [s.replace(".txt", "") for s in sources],
                    "modalities_used": list(mods_used),
                }
            except Exception:
                return {"answer": "", "evidence_files": [], "modalities_used": []}

        t4_out = [None] * len(gold["t4"])
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_t4_one, q): i for i, q in enumerate(gold["t4"])}
            for fut in as_completed(futs):
                t4_out[futs[fut]] = fut.result()
        results["t4"] = t4_out

    # ── T5: Lifecycle Maintenance (C2: CRUD + consistency) ──────────
    if "T5" in supports and "t5" in gold:
        t5_gold = gold["t5"]
        crud_probes = t5_gold.get("crud_probes", [])
        refresh_probes = t5_gold.get("refresh_probes", [])

        # Part A: CRUD — sequential (C2 lifecycle operations)
        log.info("[KC] T5A: %d CRUD probes (C2 lifecycle, sequential)...", len(crud_probes))
        crud_out = []
        for probe in crud_probes:
            try:
                fid = probe.get("file_id", "")
                op = probe.get("type", "query")

                if op == "delete" and fid in id_to_filename:
                    fname = id_to_filename[fid]
                    kc.delete_source(fname)
                    log.debug("[KC] T5: deleted %s", fname)
                elif op == "add":
                    pass  # add ops use NEW_FILE, already in KB

                question = probe.get("followup_question", "")
                result = kc.query(question, top_k=5)
                crud_out.append({"answer": result.get("answer", "")})
            except Exception:
                crud_out.append({"answer": ""})

        # Part B: Fact-change refresh
        log.info("[KC] T5B: %d refresh probes (parallel)...", len(refresh_probes))
        client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_BASE_URL)

        def _t5_refresh(probe):
            try:
                result = kc.query(probe.get("question", ""), top_k=5)
                ctx = result.get("answer", "")
                answer = _llm_call(
                    client,
                    f"Fact changed: '{probe.get('updated_fact', '')}'\n\n"
                    f"Current KB answer: {ctx}\n\n"
                    f"Q: {probe.get('question', '')}\nUpdated answer:",
                    max_tokens=200,
                )
                return {"updated_answer": answer}
            except Exception:
                return {"updated_answer": ""}

        refresh_out = [None] * len(refresh_probes)
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_t5_refresh, p): i for i, p in enumerate(refresh_probes)}
            for fut in as_completed(futs):
                refresh_out[futs[fut]] = fut.result()

        results["t5"] = {"crud_answers": crud_out, "refresh_answers": refresh_out}

    # ── T6: Knowledge Discovery (C3: research agent) ────────────────
    if "T6" in supports and "t6" in gold:
        topics = gold["t6"]
        log.info("[KC] T6: discovering knowledge for %d topics (C3 research)...", len(topics))
        client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_BASE_URL)

        def _t6_one(topic_obj):
            try:
                topic = topic_obj.get("topic", "")
                # Use research agent for gap analysis
                research_result = kc.research(topic)

                # Parse known/gaps/suggestions from research output
                known = []
                gaps = research_result.get("gaps", [])
                suggestions = research_result.get("suggestions", [])

                # Extract "known" entities via a structured LLM call on the research output
                knowledge_text = research_result.get("current_knowledge", "")
                if knowledge_text:
                    raw = _llm_call(
                        client,
                        f"Based on this KB analysis:\n{knowledge_text[:1500]}\n\n"
                        f"List the key entities/concepts the KB covers about '{topic}'.\n"
                        f"Reply ONLY JSON: {{\"known\": [\"entity1\", \"entity2\", ...]}}",
                        max_tokens=300,
                    )
                    parsed = _parse_json(raw)
                    if isinstance(parsed, dict):
                        known = parsed.get("known", [])

                return {
                    "topic": topic,
                    "known": known,
                    "gaps": gaps,
                    "suggestions": suggestions,
                }
            except Exception:
                pass
            return {"topic": topic_obj.get("topic", ""), "known": [], "gaps": [], "suggestions": []}

        t6_out = [None] * len(topics)
        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(_t6_one, t): i for i, t in enumerate(topics)}
            for fut in as_completed(futs):
                t6_out[futs[fut]] = fut.result()
        results["t6"] = t6_out

    # ── Cleanup ──────────────────────────────────────────────────────
    log.info("[KC] cleaning up temp dirs...")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(storage_dir, ignore_errors=True)

    log.info("[KC] done for %s", persona)
    return results
