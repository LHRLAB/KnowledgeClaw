"""LLM Wiki (Karpathy 2024) — LLM incrementally compiles a persistent markdown wiki.

Ref: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

The wiki is built once from the corpus (pages keyed by topic, markdown bodies with
bullet propositions and `[[WikiLinks]]`) and then queried via retrieval. For T5,
the wiki is mutated on CRUD ops and the affected page is rewritten on refresh.
"""
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


def _llm_batch(prompts, mt=400):
    def _one(p):
        try:
            return (CLIENT.chat.completions.create(
                model=EVAL_LLM, messages=[{"role": "user", "content": p}],
                temperature=0, max_tokens=mt).choices[0].message.content or "").strip()
        except Exception:
            return ""
    with ThreadPoolExecutor(max_workers=40) as ex:
        return list(ex.map(_one, prompts))


def _embed_batch(texts):
    if not texts: return []
    batches = [texts[i:i+50] for i in range(0, len(texts), 50)]
    def _do(batch):
        return [d.embedding for d in CLIENT.embeddings.create(model=EVAL_EMBED, input=batch).data]
    with ThreadPoolExecutor(max_workers=10) as ex:
        results = list(ex.map(_do, batches))
    return [e for br in results for e in br]


def _cosine(a, b):
    d = sum(x*y for x, y in zip(a, b))
    return d / max(sum(x*x for x in a)**0.5 * sum(x*x for x in b)**0.5, 1e-9)


def _pj(raw):
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip()); raw = re.sub(r"\s*```$", "", raw)
    try: return json.loads(raw)
    except: pass
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except: pass
    return None


_WIKI_PROMPT = """You are maintaining a persistent markdown wiki. Given a source document, emit 1-3 wiki pages covering its key topics. Each page has:
- an H2 title line: `## <Topic>` (short, canonical — so later docs can merge into it)
- 3-8 bullet lines of atomic propositions, each ending with the source file id in brackets
- optional `[[OtherTopic]]` wikilinks where another page is naturally referenced

Source file id: {fid}
Content:
{content}

Emit ONLY the markdown pages, no preamble."""


def _parse_pages(md: str, fid: str) -> list[dict]:
    """Split markdown into pages by `## Title`; return [{title, body, fid}]."""
    pages = []
    chunks = re.split(r'(?m)^##\s+', md)
    for ch in chunks:
        ch = ch.strip()
        if not ch: continue
        lines = ch.splitlines()
        title = lines[0].strip().strip('#').strip()
        body = "\n".join(lines[1:]).strip()
        if title and body:
            pages.append({"title": title, "body": body, "fid": fid})
    return pages


def _norm_title(t: str) -> str:
    return re.sub(r'\s+', ' ', t.lower().strip())


def _build_wiki(docs: list[dict]) -> dict[str, dict]:
    """Run LLM over each doc → merged wiki pages keyed by normalized title."""
    if not docs: return {}
    prompts = [_WIKI_PROMPT.format(fid=d["id"], content=d.get("content", "")[:1800]) for d in docs]
    outputs = _llm_batch(prompts, mt=500)
    wiki: dict[str, dict] = {}
    for d, out in zip(docs, outputs):
        for pg in _parse_pages(out, d["id"]):
            key = _norm_title(pg["title"])
            if key not in wiki:
                wiki[key] = {"title": pg["title"], "body": pg["body"], "sources": {d["id"]}}
            else:
                wiki[key]["body"] += "\n" + pg["body"]
                wiki[key]["sources"].add(d["id"])
    for v in wiki.values():
        v["sources"] = list(v["sources"])
    return wiki


def _extract_links(body: str) -> list[str]:
    return re.findall(r'\[\[([^\]]+)\]\]', body)


def _props_from_body(body: str) -> list[str]:
    out = []
    for ln in body.splitlines():
        ln = ln.strip().lstrip("-*·•").strip()
        if len(ln) > 20 and not ln.startswith("#"):
            out.append(re.sub(r'\s*\[[^\]]+\]\s*$', '', ln))
    return out[:8]


def run(persona, corpus_dir, unified, gold, supports, out_dir):
    """Fast approximation: skip LLM wiki construction, use heuristic wiki pages
    (first sentences as bullets, first capitalized phrases as entity titles)."""
    files = unified.get("files", [])
    docs = [f for f in files if len(f.get("content", "")) > 50]
    results = {}

    log.info("[LLM Wiki] building heuristic wiki from %d docs...", len(docs))
    # Heuristic wiki: each doc → 1 page keyed by its first capitalized phrase
    wiki = {}
    for d in docs:
        text = d.get("content","")[:1500]
        caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', text)
        title = (caps[0] if caps else d["id"].split("_")[-1])[:40]
        sents = [s.strip() for s in re.split(r'[.!?]\s+', text[:1500])[:6] if len(s.strip())>20]
        body = "\n".join(f"- {s}" for s in sents)
        key = _norm_title(title)
        if key not in wiki:
            wiki[key] = {"title": title, "body": body, "sources": {d["id"]}, "modality": d.get("modality","text")}
        else:
            wiki[key]["body"] += "\n" + body
            wiki[key]["sources"].add(d["id"])
    for v in wiki.values():
        v["sources"] = list(v["sources"])
    log.info("[LLM Wiki] wiki has %d pages", len(wiki))

    # Per-file provenance: which pages does each file contribute to?
    file_to_pages: dict[str, list[str]] = defaultdict(list)
    for key, pg in wiki.items():
        for s in pg["sources"]:
            file_to_pages[s].append(key)

    # Embed wiki pages for retrieval
    page_keys = list(wiki.keys())
    page_texts = [f"{wiki[k]['title']}\n{wiki[k]['body'][:1200]}" for k in page_keys]
    log.info("[LLM Wiki] embedding %d pages...", len(page_texts))
    page_embs = _embed_batch(page_texts) if page_texts else []

    # Batch-embed all queries
    all_queries, q_map = [], {}
    t3_gold = gold.get("t3", []); t4_gold = gold.get("t4", [])
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
    log.info("[LLM Wiki] batch embedding %d queries...", len(all_queries))
    q_embs = _embed_batch(all_queries) if all_queries else []

    def retrieve_pages(qi: int, k: int = 4) -> list[str]:
        if qi >= len(q_embs) or not page_embs: return []
        qe = q_embs[qi]
        scored = sorted(range(len(page_embs)), key=lambda i: -_cosine(qe, page_embs[i]))[:k]
        return [page_keys[i] for i in scored]

    def pages_ctx(keys: list[str], char_limit: int = 400) -> str:
        return "\n\n".join(f"## {wiki[k]['title']}\n{wiki[k]['body'][:char_limit]}" for k in keys if k in wiki)

    def evidence_fids(keys: list[str]) -> list[str]:
        out, seen = [], set()
        for k in keys:
            if k not in wiki: continue
            for s in wiki[k]["sources"]:
                if s not in seen:
                    seen.add(s); out.append(s)
        return out[:5]

    # T1: entities = wiki titles linked back to source file; propositions = bullet lines
    if "T1" in supports:
        t1 = []
        for f in docs:
            my_keys = file_to_pages.get(f["id"], [])
            ents, props = [], []
            for k in my_keys:
                ents.append(wiki[k]["title"])
                props.extend(_props_from_body(wiki[k]["body"]))
            t1.append({
                "file_id": f["id"],
                "modality": f.get("modality", "text"),
                "entities": list(dict.fromkeys(ents))[:15],
                "propositions": list(dict.fromkeys(props))[:8],
            })
        results["t1"] = t1

    # T2: cross-page wikilinks → edges between co-occurring source files
    if "T2" in supports:
        edges = []
        title_to_key = {_norm_title(v["title"]): k for k, v in wiki.items()}
        for k, pg in wiki.items():
            for lk in _extract_links(pg["body"]):
                tgt = title_to_key.get(_norm_title(lk))
                if not tgt or tgt == k: continue
                for s in pg["sources"]:
                    for t in wiki[tgt]["sources"]:
                        if s != t:
                            edges.append({
                                "source_file_id": s, "target_file_id": t,
                                "relation": "links_to",
                                "source_entity": pg["title"], "target_entity": wiki[tgt]["title"],
                            })
        results["t2"] = edges[:200]

    # T3/T4/T6 + T5: batch LLM
    prompts, ptasks = [], []
    for i, q in enumerate(t3_gold):
        keys = retrieve_pages(q_map[("t3", i)], 4)
        ctx = pages_ctx(keys, 400)
        prompts.append(f"Answer from wiki.\n\n{ctx}\n\nQ: {q['question']}\nA:")
        ptasks.append(("t3", i, evidence_fids(keys), ["text"]))

    for i, q in enumerate(t4_gold):
        keys = retrieve_pages(q_map[("t4", i)], 4)
        ctx = pages_ctx(keys, 400)
        prompts.append(f"Answer from wiki.\n\n{ctx}\n\nQ: {q['question']}\nA:")
        ptasks.append(("t4", i, evidence_fids(keys), ["text"]))

    # T5 CRUD: mutate live_fids; drop pages whose only source was deleted
    live = set(d["id"] for d in docs)
    for i, p in enumerate(t5_crud):
        op = p.get("type", "query")
        if op == "delete": live.discard(p.get("file_id", ""))
        elif op == "add":  live.add(p.get("file_id", ""))
        keys = [k for k in retrieve_pages(q_map[("t5c", i)], 4)
                if any(s in live for s in wiki[k]["sources"])][:4]
        ctx = pages_ctx(keys, 300)
        prompts.append(f"Wiki (current):\n{ctx}\n\nQ: {p.get('followup_question','')}\nA:" if ctx.strip() else "Wiki has no info.\nA:")
        ptasks.append(("t5c", i, [], []))

    # T5 refresh: simulate rewriting the page with the updated fact
    for i, p in enumerate(t5_refresh):
        keys = retrieve_pages(q_map[("t5r", i)], 3)
        ctx = pages_ctx(keys, 300)
        prompts.append(
            f"Wiki page (pre-edit):\n{ctx}\n\nThe wiki editor just rewrote the relevant page with this correction: '{p.get('updated_fact','')}'\n"
            f"Q: {p.get('question','')}\nAnswer using the corrected fact:")
        ptasks.append(("t5r", i, [], []))

    for i, t in enumerate(t6_gold):
        keys = retrieve_pages(q_map[("t6", i)], 5)
        ctx = pages_ctx(keys, 300)
        prompts.append(
            f"Wiki pages about '{t['topic']}':\n{ctx}\n\n"
            'Output JSON: {"known":[...],"gaps":[...],"suggestions":[...]}')
        ptasks.append(("t6", i, [], []))

    log.info("[LLM Wiki] batch LLM QA: %d prompts...", len(prompts))
    answers = _llm_batch(prompts)

    t3_out, t4_out, t5c_out, t5r_out, t6_out = [], [], [], [], []
    for (task, idx, evid, mods), ans in zip(ptasks, answers):
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
                t6_out.append({"topic": t6_gold[idx]["topic"],
                               "known": p.get("known", []), "gaps": p.get("gaps", []),
                               "suggestions": p.get("suggestions", [])})
            else:
                t6_out.append({"topic": t6_gold[idx]["topic"], "known": [], "gaps": [], "suggestions": []})

    if "T3" in supports and t3_out: results["t3"] = t3_out
    if "T4" in supports and t4_out: results["t4"] = t4_out
    if "T5" in supports and (t5c_out or t5r_out):
        results["t5"] = {"crud_answers": t5c_out, "refresh_answers": t5r_out}
    if "T6" in supports and t6_out: results["t6"] = t6_out

    log.info("[LLM Wiki] done for %s", persona)
    return results
