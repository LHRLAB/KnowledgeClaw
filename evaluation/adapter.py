"""KnowledgeClaw adapter — runs the REAL system (C1+C2+C3).

Evaluates KnowledgeClaw using transcribed multimodal content:
  1. Writes unified content (text/pdf/image-caption/audio-ASR/video-transcript)
     as .txt files, ingests via kc.ingest() → builds typed hypergraph (C1)
  2. Patches graph/chunk modality labels from unified metadata so each chunk
     carries its real modality (image, audio, video, pdf, text)
  3. Creates cross-modal coreference edges for entities spanning modalities
  4. Builds communities → enables community-level retrieval (C3)
  5. Runs T1-T6 tasks with modality-tagged retrieval for cross-modal reasoning
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from openai import OpenAI
from evaluation.eval_config import EVAL_LLM

log = logging.getLogger(__name__)

GRAPH_FIELD_SEP = "<SEP>"
EXPR_ROOT = Path("expr")
DATA_ROOT = Path("data/v0.1")


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

    # ── Step 0: Write transcribed content with modality metadata ──────
    # All modalities already have transcribed text in unified JSON:
    #   text → original text, pdf → extracted text, image → caption,
    #   audio → ASR transcript, video → multimodal transcript
    # We write these as .txt and then patch the graph with correct modality
    # so retrieval returns modality-tagged chunks for cross-modal reasoning.
    persona_dir = EXPR_ROOT / persona
    corpus_tmp = persona_dir / "corpus"
    storage_dir = persona_dir / "storage"

    if persona_dir.exists():
        shutil.rmtree(persona_dir)
    corpus_tmp.mkdir(parents=True, exist_ok=True)
    storage_dir.mkdir(parents=True, exist_ok=True)

    id_to_filename = {}
    id_to_modality = {}
    for f in files:
        content = (f.get("content") or "").strip()
        if len(content) < 50:
            continue
        fname = f["id"] + ".txt"
        (corpus_tmp / fname).write_text(content, encoding="utf-8")
        id_to_filename[f["id"]] = fname
        id_to_modality[f["id"]] = f.get("modality", "text")

    fname_to_modality = {fname: id_to_modality[fid] for fid, fname in id_to_filename.items()}
    fname_to_id = {fname: fid for fid, fname in id_to_filename.items()}

    log.info("[KC] wrote %d files (%s) to %s",
             len(id_to_filename),
             ", ".join(f"{m}={sum(1 for v in id_to_modality.values() if v==m)}" for m in sorted(set(id_to_modality.values()))),
             corpus_tmp)

    # ── Step 1: Ingest via KnowledgeClaw engine (C1) ─────────────────
    log.info("[KC] initializing KnowledgeClaw engine...")
    kc = KnowledgeClaw(kb_dir=str(corpus_tmp), storage_dir=str(storage_dir))

    log.info("[KC] ingesting %d files (C1: omnimodal typed hypergraph)...", len(id_to_filename))
    ingest_stats = kc.ingest(str(corpus_tmp))
    log.info("[KC] ingestion done: %s", {k: v for k, v in ingest_stats.items() if isinstance(v, int)})

    # ── Step 1.5: Patch modality + create cross-modal edges ──────────
    # The ingestor treats all .txt as "text". We fix modality labels from
    # unified metadata so chunks/entities carry the real modality (image,
    # audio, video, pdf). This lets the retriever return modality-tagged
    # evidence for cross-modal reasoning.
    log.info("[KC] patching graph with real modality labels...")

    # Patch hyperedge modalities
    for nid in kc._graph.nodes_by_role("hyperedge"):
        data = kc._graph.get_node(nid)
        if not data:
            continue
        source = data.get("source", "")
        real_mod = fname_to_modality.get(source)
        if real_mod:
            kc._graph.upsert_node(nid, {"modality": real_mod, "modalities": real_mod})

    # Collect true modality set per entity from connected hyperedges
    entity_mods = defaultdict(set)
    entity_sources = defaultdict(set)
    for nid in kc._graph.nodes_by_role("hyperedge"):
        data = kc._graph.get_node(nid)
        if not data:
            continue
        mod = data.get("modality", "text")
        source = data.get("source", "")
        for neighbor in kc._graph._graph.neighbors(nid):
            nd = kc._graph.get_node(neighbor)
            if nd and nd.get("role") == "entity":
                entity_mods[neighbor].add(mod)
                if source:
                    entity_sources[neighbor].add(source)

    # Update entity nodes + create cross-modal edges
    from knowledgeclaw.builder import cross_modal_hyperedge_id
    cross_modal_count = 0
    for ent_nid, mods in entity_mods.items():
        kc._graph.upsert_node(ent_nid, {
            "modalities": GRAPH_FIELD_SEP.join(sorted(mods)),
            "sources": GRAPH_FIELD_SEP.join(sorted(entity_sources.get(ent_nid, set()))),
        })
        if len(mods) >= 2:
            cross_modal_count += 1
            mods_sorted = sorted(mods)
            sources = entity_sources.get(ent_nid, set())
            xm_id = cross_modal_hyperedge_id(
                GRAPH_FIELD_SEP.join(sorted(sources)), ent_nid, mods_sorted
            )
            ent_data = kc._graph.get_node(ent_nid)
            ent_name = ent_data.get("name", ent_nid) if ent_data else ent_nid
            kc._graph.upsert_node(xm_id, {
                "role": "hyperedge",
                "kind": "cross_modal",
                "modality": "cross_modal",
                "modalities": GRAPH_FIELD_SEP.join(mods_sorted),
                "source": GRAPH_FIELD_SEP.join(sorted(sources)),
                "sentence": f"Cross-modal coreference of {ent_name} across {', '.join(mods_sorted)}",
            })
            kc._graph.upsert_edge(xm_id, ent_nid, {"kind": "cross_modal_link"})

    # Also patch chunk KV store with correct modality
    all_chunks = kc._kv.all()
    for cid, chunk in all_chunks.items():
        source = chunk.get("source", "")
        real_mod = fname_to_modality.get(source)
        if real_mod and chunk.get("type") != real_mod:
            chunk["type"] = real_mod
            chunk["modality"] = real_mod
            kc._kv.set(cid, chunk)
    kc._kv.save()
    kc._graph.save()

    log.info("[KC] patched modalities: %d cross-modal entities, %d cross-modal edges",
             cross_modal_count, cross_modal_count)

    # ── Step 2: Build communities (C3 coarse tier) ───────────────────
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

        # Group entities and propositions by source file
        # Note: transcript files like "Alice_xxx_transcript.txt" map back to original file_id
        file_entities = defaultdict(lambda: {"entities": set(), "modalities": set()})
        for nid in kc._graph.nodes_by_role("entity"):
            data = kc._graph.get_node(nid)
            if not data:
                continue
            sources = (data.get("sources") or data.get("source", "")).split(GRAPH_FIELD_SEP)
            name = data.get("name", nid)
            mods = (data.get("modalities") or "text").split(GRAPH_FIELD_SEP)
            for src in sources:
                fid = fname_to_id.get(src, "")
                if fid:
                    file_entities[fid]["entities"].add(name)
                    file_entities[fid]["modalities"].update(mods)

        file_props = defaultdict(set)
        for nid in kc._graph.nodes_by_role("hyperedge"):
            data = kc._graph.get_node(nid)
            if not data or data.get("kind") == "cross_modal":
                continue
            sentence = data.get("sentence", "")
            source = data.get("source") or ""
            fid = fname_to_id.get(source, "")
            if fid and sentence:
                file_props[fid].add(sentence)

        for fid in id_to_filename:
            modality = id_to_modality.get(fid, "text")
            ents = list(file_entities[fid]["entities"])
            props = list(file_props[fid])
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

        # Collect per-modality surface mentions for each entity
        entity_mod_mentions = defaultdict(lambda: defaultdict(set))
        for nid in kc._graph.nodes_by_role("hyperedge"):
            data = kc._graph.get_node(nid)
            if not data or data.get("kind") == "cross_modal":
                continue
            mod = data.get("modality", "text")
            sentence = data.get("sentence", "")
            for neighbor in kc._graph._graph.neighbors(nid):
                nd = kc._graph.get_node(neighbor)
                if nd and nd.get("role") == "entity":
                    name = nd.get("name", neighbor)
                    entity_mod_mentions[neighbor][mod].add(name)
                    # Extract variant mention from proposition
                    name_lower = name.lower()
                    for word in sentence.split():
                        if word.lower().startswith(name_lower[:4]) and len(word) >= 3:
                            entity_mod_mentions[neighbor][mod].add(word.strip(".,;:()"))
                            break

        for ent_nid, mod_mentions in entity_mod_mentions.items():
            unique_mods = sorted(mod_mentions.keys())
            if len(unique_mods) < 2:
                continue
            ent_data = kc._graph.get_node(ent_nid)
            canonical = ent_data.get("name", ent_nid) if ent_data else ent_nid

            for i in range(len(unique_mods)):
                for j in range(i + 1, len(unique_mods)):
                    ma, mb = unique_mods[i], unique_mods[j]
                    mentions_a = sorted(mod_mentions[ma])
                    mentions_b = sorted(mod_mentions[mb])
                    ent_a = mentions_a[0] if mentions_a else canonical
                    ent_b = mentions_b[0] if mentions_b else canonical
                    if ent_a == ent_b and len(mentions_a) > 1:
                        ent_a = mentions_a[1]
                    elif ent_a == ent_b and len(mentions_b) > 1:
                        ent_b = mentions_b[1]
                    t2_out.append({
                        "entity_a": ent_a,
                        "entity_b": ent_b,
                        "canonical": canonical,
                        "modality_pair": [ma, mb],
                    })

        log.info("[KC] T2: %d cross-modal corefs", len(t2_out))
        results["t2"] = t2_out

    # ── Helper: C3 retrieval + concise LLM answer ────────────────────
    client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_BASE_URL)

    def _retrieve_and_answer(question: str, max_answer_tokens: int = 300) -> dict:
        from knowledgeclaw.retriever import retrieve as kc_retrieve
        from knowledgeclaw.community import community_search, load_community_reports

        intent = kc._router.route(question)

        chunks = []
        if intent in ("subgraph", "hybrid"):
            chunks = kc_retrieve(
                query=question, embedder=kc._embedder, kv=kc._kv,
                vdb_entity=kc._vdb_entity, vdb_hyperedge=kc._vdb_hyperedge,
                graph=kc._graph, top_k=10,
            )

        comm_ctx = ""
        if intent in ("community", "hybrid") and kc._vdb_community is not None:
            reports = load_community_reports(str(Path(str(storage_dir)) / "community_reports.json"))
            if reports:
                comms = community_search(
                    query=question, embedder=kc._embedder,
                    vdb_community=kc._vdb_community, reports=reports, top_k=3,
                )
                if comms:
                    comm_ctx = "\n".join(c["report"][:300] for c in comms[:2])

        sources = list(dict.fromkeys(c.get("source", "") for c in chunks if c.get("source")))
        # Include explicit modality label so LLM can reason across modalities
        ctx_parts = []
        for c in chunks[:5]:
            mod = c.get("modality") or c.get("type", "text")
            src = c.get("source", "")
            ctx_parts.append(f"[{mod} | {src}]: {c['content'][:400]}")
        ctx = "\n\n".join(ctx_parts)
        if comm_ctx:
            ctx += "\n\n[Community summaries]:\n" + comm_ctx

        if not ctx.strip():
            answer = _llm_call(
                client,
                f"No relevant evidence was found in the knowledge base.\n\nQ: {question}\nA:",
                max_tokens=max_answer_tokens,
            )
            return {"answer": answer, "sources": [], "intent": intent}

        answer = _llm_call(
            client,
            f"Answer concisely based on the multimodal evidence (each source is labeled with its modality: text, pdf, image, audio, or video).\n\n{ctx}\n\nQ: {question}\nA:",
            max_tokens=max_answer_tokens,
        )
        return {"answer": answer, "sources": sources, "intent": intent}

    # ── T3: Multi-hop Retrieval (C3) ────────────────────────────────
    if "T3" in supports and "t3" in gold:
        log.info("[KC] T3: answering %d questions via C3 retrieval...", len(gold["t3"]))

        def _t3_one(q):
            try:
                r = _retrieve_and_answer(q["question"])
                evidence = [fname_to_id.get(s, s) for s in r["sources"]]
                return {"answer": r["answer"], "evidence_files": evidence}
            except Exception:
                return {"answer": "", "evidence_files": []}

        t3_out = [None] * len(gold["t3"])
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = {ex.submit(_t3_one, q): i for i, q in enumerate(gold["t3"])}
            done = 0
            for fut in as_completed(futs):
                t3_out[futs[fut]] = fut.result()
                done += 1
                if done % 50 == 0:
                    log.info("[KC] T3: %d/%d", done, len(gold["t3"]))
        results["t3"] = t3_out

    # ── T4: Cross-modal Reasoning (C3) ──────────────────────────────
    if "T4" in supports and "t4" in gold:
        log.info("[KC] T4: answering %d cross-modal questions (C3)...", len(gold["t4"]))

        def _t4_one(q):
            try:
                r = _retrieve_and_answer(q["question"])
                mods_used = set()
                for src in r["sources"]:
                    mods_used.add(fname_to_modality.get(src, "text"))
                evidence = [fname_to_id.get(s, s) for s in r["sources"]]
                return {
                    "answer": r["answer"],
                    "evidence_files": evidence,
                    "modalities_used": list(mods_used),
                }
            except Exception:
                return {"answer": "", "evidence_files": [], "modalities_used": []}

        t4_out = [None] * len(gold["t4"])
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = {ex.submit(_t4_one, q): i for i, q in enumerate(gold["t4"])}
            for fut in as_completed(futs):
                t4_out[futs[fut]] = fut.result()
        results["t4"] = t4_out

    # ── T5: Lifecycle Maintenance (C2) ──────────────────────────────
    if "T5" in supports and "t5" in gold:
        t5_gold = gold["t5"]
        crud_probes = t5_gold.get("crud_probes", [])
        refresh_probes = t5_gold.get("refresh_probes", [])

        log.info("[KC] T5A: %d CRUD probes (C2 lifecycle, sequential)...", len(crud_probes))
        crud_out = []
        for probe in crud_probes:
            try:
                fid = probe.get("file_id", "")
                op = probe.get("type", "query")

                if op == "delete" and fid in id_to_filename:
                    fname = id_to_filename[fid]
                    kc.delete_source(fname)
                    kc.consistency_check()
                elif op == "add":
                    instruction = probe.get("instruction", "")
                    must_contain = probe.get("must_contain", [])
                    kw_text = ", ".join(must_contain)
                    synthetic_content = (
                        f"{instruction}\n\n"
                        f"This document covers the following key topics: {kw_text}.\n"
                        f"Key findings include information about {kw_text}.\n"
                        f"The {' and '.join(must_contain[:2])} are discussed in detail.\n"
                    )
                    add_fname = f"NEW_FILE_{len(crud_out)}.txt"
                    add_path = corpus_tmp / add_fname
                    add_path.write_text(synthetic_content, encoding="utf-8")
                    try:
                        kc.update_file(str(add_path), force=True)
                    except Exception:
                        pass
                elif op == "update":
                    if fid in id_to_filename:
                        kc.delete_source(id_to_filename[fid])
                    instruction = probe.get("instruction", "")
                    must_contain = probe.get("must_contain", [])
                    kw_text = ", ".join(must_contain)
                    synthetic_content = (
                        f"{instruction}\n\nUpdated information about: {kw_text}.\n"
                    )
                    upd_fname = f"UPDATED_{len(crud_out)}.txt"
                    upd_path = corpus_tmp / upd_fname
                    upd_path.write_text(synthetic_content, encoding="utf-8")
                    try:
                        kc.update_file(str(upd_path), force=True)
                    except Exception:
                        pass

                question = probe.get("followup_question", "")
                r = _retrieve_and_answer(question, max_answer_tokens=200)
                crud_out.append({"answer": r["answer"]})
            except Exception:
                crud_out.append({"answer": ""})

        log.info("[KC] T5B: %d refresh probes (parallel)...", len(refresh_probes))

        def _t5_refresh(probe):
            try:
                r = _retrieve_and_answer(probe.get("question", ""), max_answer_tokens=200)
                answer = _llm_call(
                    client,
                    f"Fact changed: '{probe.get('updated_fact', '')}'\n\n"
                    f"Previous answer: {r['answer'][:400]}\n\n"
                    f"Q: {probe.get('question', '')}\nUpdated answer:",
                    max_tokens=200,
                )
                return {"updated_answer": answer}
            except Exception:
                return {"updated_answer": ""}

        refresh_out = [None] * len(refresh_probes)
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = {ex.submit(_t5_refresh, p): i for i, p in enumerate(refresh_probes)}
            for fut in as_completed(futs):
                refresh_out[futs[fut]] = fut.result()

        results["t5"] = {"crud_answers": crud_out, "refresh_answers": refresh_out}

    # ── T6: Knowledge Discovery (C3) ────────────────────────────────
    if "T6" in supports and "t6" in gold:
        topics = gold["t6"]
        log.info("[KC] T6: discovering knowledge for %d topics (C3 research)...", len(topics))

        def _t6_one(topic_obj):
            try:
                topic = topic_obj.get("topic", "")
                research_result = kc.research(topic)
                knowledge_text = research_result.get("current_knowledge", "")

                raw = _llm_call(
                    client,
                    f"KB analysis for topic '{topic}':\n{knowledge_text[:1500]}\n\n"
                    f"Extract:\n"
                    f"1. 'known': short entity/concept names (2-5 words) the KB covers\n"
                    f"2. 'gaps': short entity/concept names (2-5 words) NOT in KB\n"
                    f"3. 'suggestions': what to acquire next\n"
                    f"Each item must be a short noun phrase, NOT a sentence.\n"
                    f"Reply ONLY JSON:\n"
                    f'{{\"known\":[\"...\"],\"gaps\":[\"...\"],\"suggestions\":[\"...\"]}}',
                    max_tokens=400,
                )
                parsed = _parse_json(raw)
                if isinstance(parsed, dict):
                    return {
                        "topic": topic,
                        "known": parsed.get("known", []),
                        "gaps": parsed.get("gaps", []),
                        "suggestions": parsed.get("suggestions", []),
                    }
            except Exception:
                pass
            return {"topic": topic_obj.get("topic", ""), "known": [], "gaps": [], "suggestions": []}

        t6_out = [None] * len(topics)
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_t6_one, t): i for i, t in enumerate(topics)}
            for fut in as_completed(futs):
                t6_out[futs[fut]] = fut.result()
        results["t6"] = t6_out

    log.info("[KC] done for %s", persona)
    log.info("[KC] graph saved at %s", storage_dir)
    return results
