"""KnowledgeClaw adapter — full system (C1+C2+C3).

Runs the KnowledgeClaw evaluation pipeline:
  - T1: LLM entity+proposition extraction from each file
  - T2: LLM cross-modal entity matching
  - T3: Embedding retrieval + LLM answer generation
  - T4: Same as T3 with modality tracking
  - T5: CRUD lifecycle tracking with delete support
  - T6: Knowledge discovery (known/gaps identification)
"""
from __future__ import annotations

import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config as cfg
from openai import OpenAI
from evaluation.eval_config import EVAL_LLM, EVAL_EMBED

log = logging.getLogger(__name__)

CLIENT = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_BASE_URL)
MODEL = EVAL_LLM
EMBED_MODEL = EVAL_EMBED


def _llm(prompt: str, max_tokens: int = 500) -> str:
    resp = CLIENT.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _embed(texts: list[str]) -> list[list[float]]:
    resp = CLIENT.embeddings.create(model=EMBED_MODEL, input=texts[:50])
    return [d.embedding for d in resp.data]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / max(na * nb, 1e-9)


def _parse_json(raw: str):
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        return None


def run(persona, corpus_dir, unified, gold, supports, out_dir):
    files = unified.get("files", [])
    results = {}

    # Build index: embed all file contents
    log.info("[KC] building embedding index for %d files...", len(files))
    docs = []
    texts_to_embed = []
    for f in files:
        content = (f.get("content") or "").strip()
        if len(content) < 50:
            continue
        docs.append(f)
        texts_to_embed.append(content[:2000])

    embeddings = []
    for i in range(0, len(texts_to_embed), 50):
        embeddings.extend(_embed(texts_to_embed[i:i+50]))

    def retrieve_top_k(query, k=5):
        if not embeddings:
            return []
        q_emb = _embed([query])[0]
        scored = [(i, _cosine(q_emb, e)) for i, e in enumerate(embeddings)]
        scored.sort(key=lambda x: -x[1])
        return [docs[i] for i, _ in scored[:k]]

    # ── T1: Entity + Proposition extraction (40-way parallel) ──
    if "T1" in supports:
        t1_files = [f for f in files if len(f.get("content", "")) > 50]
        log.info("[KC] T1: extracting entities for %d files (parallel)...", len(t1_files))

        def _t1_one(f):
            content = f.get("content", "")[:2000]
            try:
                raw = _llm(
                    f"Extract entities and propositions from this {f.get('modality','text')} content.\n\n"
                    f"{content[:1500]}\n\n"
                    f"Output JSON: {{\"entities\": [\"ENT1\", ...], \"propositions\": [\"prop1\", ...]}}",
                    max_tokens=600,
                )
                parsed = _parse_json(raw)
                if isinstance(parsed, dict):
                    return {"file_id": f["id"], "modality": f.get("modality", ""),
                            "entities": parsed.get("entities", [])[:15],
                            "propositions": parsed.get("propositions", [])[:8]}
            except Exception:
                pass
            return {"file_id": f["id"], "modality": f.get("modality", ""),
                    "entities": [], "propositions": []}

        t1_out = [None] * len(t1_files)
        with ThreadPoolExecutor(max_workers=40) as ex:
            futs = {ex.submit(_t1_one, f): i for i, f in enumerate(t1_files)}
            done = 0
            for fut in as_completed(futs):
                t1_out[futs[fut]] = fut.result()
                done += 1
                if done % 100 == 0:
                    log.info("[KC] T1: %d/%d", done, len(t1_files))
        results["t1"] = t1_out

    # ── T2: Cross-modal coreference (reference-free) ──
    if "T2" in supports:
        log.info("[KC] T2: cross-modal entity matching...")
        t2_out = []
        if "t1" in results:
            by_mod = {}
            for item in results["t1"]:
                mod = item.get("modality", "text")
                for e in item.get("entities", []):
                    by_mod.setdefault(mod, set()).add(e)

            mods = list(by_mod.keys())
            for i in range(len(mods)):
                for j in range(i+1, len(mods)):
                    ma, mb = mods[i], mods[j]
                    ents_a = list(by_mod[ma])[:15]
                    ents_b = list(by_mod[mb])[:15]
                    if not ents_a or not ents_b:
                        continue
                    raw = _llm(
                        f"Match entities across modalities.\n"
                        f"Modality {ma}: {ents_a}\n"
                        f"Modality {mb}: {ents_b}\n"
                        f"Output JSON: {{\"corefs\": [{{\"entity_a\": \"...\", \"entity_b\": \"...\", \"canonical\": \"...\"}}, ...]}}",
                        max_tokens=500,
                    )
                    parsed = _parse_json(raw)
                    if isinstance(parsed, dict):
                        for c in parsed.get("corefs", []):
                            c["modality_pair"] = [ma, mb]
                            t2_out.append(c)
        results["t2"] = t2_out

    # ── T3: Multi-hop retrieval (40-way parallel) ──
    if "T3" in supports and "t3" in gold:
        log.info("[KC] T3: answering %d retrieval questions (parallel)...", len(gold["t3"]))

        def _t3_one(q):
            try:
                top = retrieve_top_k(q["question"], k=5)
                context = "\n\n".join(f"[{d['id']}] ({d['modality']}): {d['content'][:600]}" for d in top)
                answer = _llm(f"Answer based on context:\n\n{context}\n\nQuestion: {q['question']}\nAnswer:", max_tokens=300)
                return {"answer": answer, "evidence_files": [d["id"] for d in top]}
            except Exception:
                return {"answer": "", "evidence_files": []}

        t3_out = [None] * len(gold["t3"])
        with ThreadPoolExecutor(max_workers=40) as ex:
            futs = {ex.submit(_t3_one, q): i for i, q in enumerate(gold["t3"])}
            done = 0
            for fut in as_completed(futs):
                t3_out[futs[fut]] = fut.result()
                done += 1
                if done % 50 == 0: log.info("[KC] T3: %d/%d", done, len(gold["t3"]))
        results["t3"] = t3_out

    # ── T4: Cross-modal reasoning (40-way parallel) ──
    if "T4" in supports and "t4" in gold:
        log.info("[KC] T4: answering %d cross-modal questions (parallel)...", len(gold["t4"]))

        def _t4_one(q):
            try:
                top = retrieve_top_k(q["question"], k=5)
                context = "\n\n".join(f"[{d['id']}] ({d['modality']}): {d['content'][:600]}" for d in top)
                answer = _llm(f"Answer using multiple modalities:\n\n{context}\n\nQuestion: {q['question']}\nAnswer:", max_tokens=300)
                mods_used = list(set(d["modality"] for d in top))
                return {"answer": answer, "evidence_files": [d["id"] for d in top], "modalities_used": mods_used}
            except Exception:
                return {"answer": "", "evidence_files": [], "modalities_used": []}

        t4_out = [None] * len(gold["t4"])
        with ThreadPoolExecutor(max_workers=40) as ex:
            futs = {ex.submit(_t4_one, q): i for i, q in enumerate(gold["t4"])}
            for fut in as_completed(futs): t4_out[futs[fut]] = fut.result()
        results["t4"] = t4_out

    # ── T5: Lifecycle Maintenance (CRUD + fact-change refresh) ──
    if "T5" in supports and "t5" in gold:
        t5_gold = gold["t5"]
        crud_probes = t5_gold.get("crud_probes", [])
        refresh_probes = t5_gold.get("refresh_probes", [])

        log.info("[KC] T5A: %d CRUD probes (sequential)...", len(crud_probes))
        kb_state = set(f["id"] for f in files)

        crud_out = []
        for probe in crud_probes:
            try:
                fid = probe.get("file_id", "")
                op = probe.get("type", "query")
                if op == "delete":
                    kb_state.discard(fid)
                elif op == "add":
                    kb_state.add(fid)

                question = probe.get("followup_question", "")
                top = retrieve_top_k(question, k=5)
                top = [d for d in top if d["id"] in kb_state][:3]
                context = "\n\n".join(f"[{d['id']}]: {d['content'][:400]}" for d in top)
                answer = _llm(f"Context:\n{context}\n\nQ: {question}\nA:", max_tokens=200)
                crud_out.append({"answer": answer})
            except Exception:
                crud_out.append({"answer": ""})

        log.info("[KC] T5B: %d refresh probes (parallel)...", len(refresh_probes))

        def _t5_refresh(probe):
            try:
                top = retrieve_top_k(probe.get("question", ""), k=3)
                ctx = "\n\n".join(f"{d['content'][:400]}" for d in top)
                answer = _llm(
                    f"Fact changed: '{probe.get('updated_fact','')}'\n\n{ctx}\n\n"
                    f"Q: {probe.get('question','')}\nUpdated answer:", max_tokens=200)
                return {"updated_answer": answer}
            except Exception:
                return {"updated_answer": ""}

        refresh_out = [None] * len(refresh_probes)
        with ThreadPoolExecutor(max_workers=40) as ex:
            futs = {ex.submit(_t5_refresh, p): i for i, p in enumerate(refresh_probes)}
            for fut in as_completed(futs): refresh_out[futs[fut]] = fut.result()

        results["t5"] = {"crud_answers": crud_out, "refresh_answers": refresh_out}

    # ── T6: Knowledge Discovery (reference-free) ──
    if "T6" in supports and "t6" in gold:
        topics = gold["t6"]
        log.info("[KC] T6: discovering knowledge for %d topics (parallel)...", len(topics))

        def _t6_one(topic_obj):
            try:
                topic = topic_obj.get("topic", "")
                top = retrieve_top_k(topic, k=10)
                context = "\n\n".join(f"{d['content'][:300]}" for d in top)
                raw = _llm(
                    f"Given this topic: '{topic}'\n\n"
                    f"Based on the following KB content:\n{context}\n\n"
                    f"Identify:\n1. 'known': short entity or concept names (2-5 words each) that the KB covers\n"
                    f"2. 'gaps': short entity or concept names (2-5 words each) NOT covered by the KB\n"
                    f"3. 'suggestions': short descriptions of what to acquire next\n\n"
                    f"IMPORTANT: Each item must be a short noun phrase (e.g. 'atrial fibrillation treatment'), NOT a full sentence.\n\n"
                    f"Output JSON: {{\"known\": [\"entity1\", ...], \"gaps\": [\"entity1\", ...], \"suggestions\": [\"...\"]}}",
                    max_tokens=500)
                parsed = _parse_json(raw)
                if isinstance(parsed, dict):
                    return {"topic": topic, "known": parsed.get("known", []),
                            "gaps": parsed.get("gaps", []), "suggestions": parsed.get("suggestions", [])}
            except Exception:
                pass
            return {"topic": topic_obj.get("topic", ""), "known": [], "gaps": [], "suggestions": []}

        t6_out = [None] * len(topics)
        with ThreadPoolExecutor(max_workers=20) as ex:
            futs = {ex.submit(_t6_one, t): i for i, t in enumerate(topics)}
            for fut in as_completed(futs): t6_out[futs[fut]] = fut.result()
        results["t6"] = t6_out

    log.info("[KC] done for %s", persona)
    return results
