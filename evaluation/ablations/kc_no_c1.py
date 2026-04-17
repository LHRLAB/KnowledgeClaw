"""KC w/o C1 — no multi-granularity retrieval. No omnimodal. Text/PDF only. T1 MCov drops, T2→0."""
from __future__ import annotations
import json, logging, re, sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import config as cfg
from openai import OpenAI
from evaluation.eval_config import EVAL_LLM, EVAL_EMBED

log = logging.getLogger(__name__)
CLIENT = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_BASE_URL)

def _llm_batch(prompts, mt=300):
    """Parallel LLM calls."""
    def _one(p):
        try:
            return (CLIENT.chat.completions.create(model=EVAL_LLM,messages=[{"role":"user","content":p}],temperature=0,max_tokens=mt).choices[0].message.content or "").strip()
        except: return ""
    with ThreadPoolExecutor(max_workers=40) as ex:
        return list(ex.map(_one, prompts))

def _embed_batch(texts):
    if not texts: return []
    batches = [texts[i:i+50] for i in range(0, len(texts), 50)]
    def _do(batch):
        return [d.embedding for d in CLIENT.embeddings.create(model=EVAL_EMBED, input=batch).data]
    from concurrent.futures import ThreadPoolExecutor as TPE
    with TPE(max_workers=10) as ex:
        results = list(ex.map(_do, batches))
    return [e for batch_result in results for e in batch_result]

def _cosine(a,b):
    d=sum(x*y for x,y in zip(a,b)); return d/max(sum(x*x for x in a)**0.5*sum(x*x for x in b)**0.5,1e-9)

def _pj(raw):
    raw=re.sub(r"^```(?:json)?\s*","",raw.strip()); raw=re.sub(r"\s*```$","",raw)
    try: return json.loads(raw)
    except: pass
    m=re.search(r'\{.*\}',raw,re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    return None

def run(persona, corpus_dir, unified, gold, supports, out_dir):
    files = unified.get("files", [])
    docs = [f for f in files if f.get("modality","text") in ("text","pdf") and len(f.get("content","")) > 50]
    results = {}

    # Step 1: Embed all docs (one batch call)
    log.info("[KC w/o C1] embedding %d docs...", len(docs))
    doc_embs = _embed_batch([d["content"][:2000] for d in docs])

    # Step 2: Collect ALL queries upfront and batch embed them
    all_queries = []
    q_map = {}  # (task, index) → position in all_queries

    t3_gold = gold.get("t3", [])
    t4_gold = gold.get("t4", [])
    t5_crud = gold.get("t5", {}).get("crud_probes", [])
    t5_refresh = gold.get("t5", {}).get("refresh_probes", [])
    t6_gold = gold.get("t6", [])

    for i, q in enumerate(t3_gold):
        q_map[("t3", i)] = len(all_queries); all_queries.append(q["question"])
    for i, q in enumerate(t4_gold):
        q_map[("t4", i)] = len(all_queries); all_queries.append(q["question"])
    for i, p in enumerate(t5_crud):
        q_map[("t5c", i)] = len(all_queries); all_queries.append(p.get("followup_question", ""))
    for i, p in enumerate(t5_refresh):
        q_map[("t5r", i)] = len(all_queries); all_queries.append(p.get("question", ""))
    for i, t in enumerate(t6_gold):
        q_map[("t6", i)] = len(all_queries); all_queries.append(t.get("topic", ""))

    log.info("[KC w/o C1] batch embedding %d queries...", len(all_queries))
    q_embs = _embed_batch(all_queries) if all_queries else []

    def retrieve(query_idx, k=3):
        if query_idx >= len(q_embs): return []
        qe = q_embs[query_idx]
        scored = sorted(enumerate(doc_embs), key=lambda x: -_cosine(qe, x[1]))[:k]
        return [docs[i] for i, _ in scored]

    # T1: fast keyword extraction (zero LLM)
    if "T1" in supports:
        def _kw(text):
            words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text[:2000])
            words += re.findall(r'\b[A-Z]{2,}\b', text[:2000])
            return [w for w, _ in Counter(words).most_common(15) if len(w) >= 3]
        def _props(text):
            return [s.strip() for s in re.split(r'[.!?]\s+', text[:2000])[:8] if len(s.strip()) > 20]
        results["t1"] = [{"file_id":f["id"],"modality":"text",
            "entities":_kw(f.get("content","")),"propositions":_props(f.get("content",""))} for f in docs]

    # T2: empty (C1 ablated — text only, no cross-modal)
    if "T2" in supports:
        results["t2"] = []

    # T3-T6: batch LLM (all prompts collected, then parallel)
    prompts = []
    prompt_tasks = []  # (task, index)

    for i in range(len(t3_gold)):
        top = retrieve(q_map[("t3",i)])
        ctx = "\n\n".join(f"[{d['id']}]: {d['content'][:400]}" for d in top)
        prompts.append(f"Answer concisely.\n\n{ctx}\n\nQ: {t3_gold[i]['question']}\nA:")
        prompt_tasks.append(("t3", i, [d["id"] for d in top], ["text"]))

    for i in range(len(t4_gold)):
        top = retrieve(q_map[("t4",i)])
        ctx = "\n\n".join(f"[{d['id']}]: {d['content'][:400]}" for d in top)
        prompts.append(f"Answer concisely.\n\n{ctx}\n\nQ: {t4_gold[i]['question']}\nA:")
        prompt_tasks.append(("t4", i, [d["id"] for d in top], ["text"]))

    kb = set(d["id"] for d in docs)
    for i in range(len(t5_crud)):
        op = t5_crud[i].get("type","query")
        if op == "delete": kb.discard(t5_crud[i].get("file_id",""))
        elif op == "add": kb.add(t5_crud[i].get("file_id",""))
        top = [d for d in retrieve(q_map[("t5c",i)]) if d["id"] in kb][:3]
        ctx = "\n".join(d["content"][:300] for d in top)
        prompts.append(f"{ctx}\n\nQ: {t5_crud[i].get('followup_question','')}\nA:" if ctx.strip() else "No info.\nA:")
        prompt_tasks.append(("t5c", i, [], []))

    for i in range(len(t5_refresh)):
        top = retrieve(q_map[("t5r",i)], 3)
        ctx = "\n".join(d["content"][:300] for d in top)
        prompts.append(f"Fact changed: '{t5_refresh[i].get('updated_fact','')}'\n{ctx}\nQ: {t5_refresh[i].get('question','')}\nAnswer:")
        prompt_tasks.append(("t5r", i, [], []))

    for i in range(len(t6_gold)):
        top = retrieve(q_map[("t6",i)], 5)
        ctx = "\n".join(d["content"][:300] for d in top)
        prompts.append(f"KB about '{t6_gold[i]['topic']}':\n{ctx}\n\nJSON: {{\"known\":[],\"gaps\":[],\"suggestions\":[]}}")
        prompt_tasks.append(("t6", i, [], []))

    log.info("[KC w/o C1] batch LLM: %d prompts...", len(prompts))
    answers = _llm_batch(prompts)

    # Assemble results
    t3_out, t4_out, t5c_out, t5r_out, t6_out = [], [], [], [], []
    for (task, idx, evid, mods), ans in zip(prompt_tasks, answers):
        if task == "t3":
            t3_out.append({"answer": ans, "evidence_files": evid})
        elif task == "t4":
            t4_out.append({"answer": ans, "evidence_files": evid, "modalities_used": mods})
        elif task == "t5c":
            t5c_out.append({"answer": ans})
        elif task == "t5r":
            t5r_out.append({"updated_answer": ans})
        elif task == "t6":
            p = _pj(ans)
            if isinstance(p, dict):
                t6_out.append({"topic": t6_gold[idx]["topic"], "known": p.get("known",[]), "gaps": p.get("gaps",[]), "suggestions": p.get("suggestions",[])})
            else:
                t6_out.append({"topic": t6_gold[idx]["topic"], "known": [], "gaps": [], "suggestions": []})

    if "T3" in supports and t3_out: results["t3"] = t3_out
    if "T4" in supports and t4_out: results["t4"] = t4_out
    if "T5" in supports and (t5c_out or t5r_out): results["t5"] = {"crud_answers": t5c_out, "refresh_answers": t5r_out}
    if "T6" in supports and t6_out: results["t6"] = t6_out

    log.info("[KC w/o C1] done for %s", persona)
    return results
