"""
KnowledgeClaw read-only agents:

  * QAAgent        — hierarchical multi-granular Q&A (C3) using the intent
                     router to choose between subgraph- and community-level
                     retrieval (or both).
  * InferenceAgent — ReACT-style multi-round inference / 推演.
  * ResearchAgent  — KB gap analysis and expansion suggestions.
  * EvalAgent      — KB health evaluation.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from knowledgeclaw.builder import GRAPH_FIELD_SEP
from knowledgeclaw.community import community_search, load_community_reports
from knowledgeclaw.retriever import retrieve
from knowledgeclaw.router import IntentRouter
from knowledgeclaw.storage import GraphStorage, JsonKVStorage, VectorStorage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_chunks(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        src = c.get("source", "unknown")
        modality = c.get("modality") or c.get("type", "text")
        content = c.get("content", "")
        parts.append(f"[chunk-{i}] (source={src}, modality={modality})\n{content}")
    return "\n\n---\n\n".join(parts)


def _fmt_communities(reports: List[Dict]) -> str:
    parts = []
    for i, r in enumerate(reports, 1):
        parts.append(
            f"[community-{i}] id={r['community_id']} size={r.get('size', 0)} "
            f"modalities={r.get('modalities', '')}\n{r.get('report', '')}"
        )
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# QAAgent — hierarchical multi-granular retrieval (C3)
# ---------------------------------------------------------------------------

class QAAgent:
    """Answers questions using whichever retrieval tier(s) the router selects."""

    SYSTEM_PROMPT = (
        "You are a precise knowledge assistant for a personal knowledge "
        "hypergraph. Answer the user's question using ONLY the provided "
        "evidence. If the evidence is insufficient, say so clearly. "
        "Cite chunk-N or community-N when you reference a fact."
    )

    def __init__(
        self,
        client: OpenAI,
        model: str,
        embedder,
        kv: JsonKVStorage,
        vdb_entity: VectorStorage,
        vdb_hyperedge: VectorStorage,
        graph: GraphStorage,
        vdb_community: Optional[VectorStorage] = None,
        community_reports_path: Optional[str] = None,
        router: Optional[IntentRouter] = None,
    ):
        self._client = client
        self._model = model
        self._embedder = embedder
        self._kv = kv
        self._vdb_entity = vdb_entity
        self._vdb_hyperedge = vdb_hyperedge
        self._graph = graph
        self._vdb_community = vdb_community
        self._community_reports_path = community_reports_path
        self._router = router or IntentRouter(client, model, use_llm=False)

    # ------------------------------------------------------------------ run

    def run(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        intent = self._router.route(question)

        chunks: List[Dict] = []
        communities: List[Dict] = []

        if intent in ("subgraph", "hybrid"):
            chunks = retrieve(
                query=question,
                embedder=self._embedder,
                kv=self._kv,
                vdb_entity=self._vdb_entity,
                vdb_hyperedge=self._vdb_hyperedge,
                graph=self._graph,
                top_k=top_k,
            )

        if intent in ("community", "hybrid") and self._vdb_community is not None:
            reports = load_community_reports(self._community_reports_path or "")
            if reports:
                communities = community_search(
                    query=question,
                    embedder=self._embedder,
                    vdb_community=self._vdb_community,
                    reports=reports,
                    top_k=max(2, top_k // 3),
                )

        if not chunks and not communities:
            return {
                "answer": "No relevant information found in the knowledge base for your question.",
                "intent": intent,
                "sources": [],
                "chunks_used": 0,
                "communities_used": 0,
            }

        evidence_parts = []
        if chunks:
            evidence_parts.append("=== SUBGRAPH EVIDENCE ===\n" + _fmt_chunks(chunks))
        if communities:
            evidence_parts.append("=== COMMUNITY EVIDENCE ===\n" + _fmt_communities(communities))
        evidence = "\n\n".join(evidence_parts)

        sources = list(dict.fromkeys(c.get("source", "") for c in chunks if c.get("source")))

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n\n"
                            f"Routed intent: {intent}\n\n"
                            f"Evidence:\n\n{evidence}\n\nAnswer:"
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("QAAgent LLM call failed: %s", exc)
            answer = f"[LLM error: {exc}]"

        return {
            "answer": answer,
            "intent": intent,
            "sources": sources,
            "chunks_used": len(chunks),
            "communities_used": len(communities),
        }


# ---------------------------------------------------------------------------
# InferenceAgent — ReACT multi-round reasoning
# ---------------------------------------------------------------------------

class InferenceAgent:
    """ReACT-style inference / 推演 over the typed hypergraph."""

    SYSTEM_PROMPT = """You are a knowledge inference engine analysing a personal knowledge hypergraph.

You have a search tool. Use it to find relevant knowledge, then synthesise deep insights.

Format your response with these XML-like tags:
<think>your reasoning about what to search next or what you have learned so far</think>
<search>a specific search query to retrieve relevant knowledge</search>

After you have gathered enough knowledge (usually 3-5 searches), produce your final insight:
<insight>your synthesised insight, prediction, or conclusion — cite specific facts from the knowledge base</insight>

Rules:
- Each <search> triggers an actual KB retrieval through the hierarchical retriever (subgraph + community).
- You may do up to {max_rounds} search rounds before you must produce <insight>.
- Always base insights on retrieved facts; note when evidence is sparse.
- Write the <insight> tag LAST, after all your thinking and searching."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        embedder,
        kv: JsonKVStorage,
        vdb_entity: VectorStorage,
        vdb_hyperedge: VectorStorage,
        graph: GraphStorage,
        vdb_community: Optional[VectorStorage] = None,
        community_reports_path: Optional[str] = None,
        router: Optional[IntentRouter] = None,
    ):
        self._client = client
        self._model = model
        self._embedder = embedder
        self._kv = kv
        self._vdb_entity = vdb_entity
        self._vdb_hyperedge = vdb_hyperedge
        self._graph = graph
        self._vdb_community = vdb_community
        self._community_reports_path = community_reports_path
        self._router = router or IntentRouter(client, model, use_llm=False)

    def run(self, topic: str, max_rounds: int = 5) -> Dict[str, Any]:
        system = self.SYSTEM_PROMPT.format(max_rounds=max_rounds)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Topic for inference / 推演: {topic}"},
        ]

        queries_used: List[str] = []
        all_sources: List[str] = []
        final_insight = ""

        for round_num in range(max_rounds + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model, messages=messages,
                    temperature=0.4, max_tokens=1500,
                )
                assistant_text = response.choices[0].message.content or ""
            except Exception as exc:
                logger.error("InferenceAgent LLM call failed: %s", exc)
                break

            messages.append({"role": "assistant", "content": assistant_text})

            insight_match = re.search(r"<insight>(.*?)</insight>", assistant_text, re.DOTALL)
            if insight_match:
                final_insight = insight_match.group(1).strip()
                break

            search_matches = re.findall(r"<search>(.*?)</search>", assistant_text, re.DOTALL)
            if not search_matches:
                if round_num >= max_rounds - 1:
                    messages.append({"role": "user",
                                     "content": "You have reached the search limit. Please produce your <insight> now."})
                    continue
                messages.append({"role": "user",
                                 "content": "Continue – either search for more information or provide your <insight>."})
                continue

            observations = []
            for query in search_matches:
                query = query.strip()
                if not query:
                    continue
                queries_used.append(query)
                hits = self._hierarchical_search(query)
                for c in hits["chunks"]:
                    src = c.get("source", "")
                    if src and src not in all_sources:
                        all_sources.append(src)

                if hits["chunks"] or hits["communities"]:
                    obs_parts = []
                    if hits["chunks"]:
                        obs_parts.append(_fmt_chunks(hits["chunks"][:3]))
                    if hits["communities"]:
                        obs_parts.append(_fmt_communities(hits["communities"][:2]))
                    observations.append(f"Search: '{query}'\n\nResults:\n" + "\n\n".join(obs_parts))
                else:
                    observations.append(f"Search: '{query}'\n\nResults: No relevant information found.")

            obs_text = "\n\n===\n\n".join(observations)
            messages.append({"role": "user",
                             "content": (f"[KB Search Results]\n\n{obs_text}\n\n"
                                         "Continue thinking. If you have enough information, produce your <insight>.")})

        if not final_insight:
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages + [{"role": "user",
                                          "content": "Based on everything retrieved so far, produce your final <insight>."}],
                    temperature=0.4, max_tokens=800,
                )
                text = response.choices[0].message.content or ""
                insight_match = re.search(r"<insight>(.*?)</insight>", text, re.DOTALL)
                final_insight = insight_match.group(1).strip() if insight_match else text.strip()
            except Exception as exc:
                final_insight = f"[Could not generate insight: {exc}]"

        return {
            "insight": final_insight,
            "queries_used": queries_used,
            "sources": all_sources,
        }

    # ------------------------------------------------------ hierarchical search

    def _hierarchical_search(self, query: str) -> Dict[str, List[Dict]]:
        intent = self._router.route(query)
        chunks: List[Dict] = []
        communities: List[Dict] = []
        if intent in ("subgraph", "hybrid"):
            chunks = retrieve(
                query=query, embedder=self._embedder, kv=self._kv,
                vdb_entity=self._vdb_entity, vdb_hyperedge=self._vdb_hyperedge,
                graph=self._graph, top_k=5,
            )
        if intent in ("community", "hybrid") and self._vdb_community is not None:
            reports = load_community_reports(self._community_reports_path or "")
            if reports:
                communities = community_search(
                    query=query, embedder=self._embedder,
                    vdb_community=self._vdb_community, reports=reports, top_k=2,
                )
        return {"chunks": chunks, "communities": communities}


# ---------------------------------------------------------------------------
# ResearchAgent — gap analysis
# ---------------------------------------------------------------------------

class ResearchAgent:
    """Autonomously analyses KB coverage of a topic and identifies gaps."""

    SYSTEM_PROMPT = """You are a research analyst for a personal knowledge hypergraph.

Given the topic and the retrieved knowledge snippets (which include both fine-grained
chunks and community-level summaries), you must:
1. Summarise what the KB already knows about the topic.
2. Identify specific knowledge gaps – what is missing or unclear.
3. Suggest 5-10 concrete search queries or sub-topics to fill those gaps.
4. Rate the current coverage: low / medium / high.

Return a structured analysis."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        embedder,
        kv: JsonKVStorage,
        vdb_entity: VectorStorage,
        vdb_hyperedge: VectorStorage,
        graph: GraphStorage,
        vdb_community: Optional[VectorStorage] = None,
        community_reports_path: Optional[str] = None,
        router: Optional[IntentRouter] = None,
    ):
        self._client = client
        self._model = model
        self._embedder = embedder
        self._kv = kv
        self._vdb_entity = vdb_entity
        self._vdb_hyperedge = vdb_hyperedge
        self._graph = graph
        self._vdb_community = vdb_community
        self._community_reports_path = community_reports_path
        self._router = router

    def run(self, topic: str, kb_dir: str = "") -> Dict[str, Any]:
        chunks = retrieve(
            query=topic, embedder=self._embedder, kv=self._kv,
            vdb_entity=self._vdb_entity, vdb_hyperedge=self._vdb_hyperedge,
            graph=self._graph, top_k=15,
        )
        communities: List[Dict] = []
        if self._vdb_community is not None and self._community_reports_path:
            reports = load_community_reports(self._community_reports_path)
            if reports:
                communities = community_search(
                    query=topic, embedder=self._embedder,
                    vdb_community=self._vdb_community, reports=reports, top_k=5,
                )

        sources = list(dict.fromkeys(c.get("source", "") for c in chunks if c.get("source")))
        evidence_parts: List[str] = []
        if chunks:
            evidence_parts.append(_fmt_chunks(chunks))
        if communities:
            evidence_parts.append(_fmt_communities(communities))
        context = "\n\n".join(evidence_parts) or "No relevant information found in the knowledge base."

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": (
                        f"Topic: {topic}\n\nRetrieved Knowledge:\n\n{context}\n\n"
                        "Provide your gap analysis."
                    )},
                ],
                temperature=0.3, max_tokens=1500,
            )
            analysis_text = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error("ResearchAgent LLM call failed: %s", exc)
            analysis_text = f"[LLM error: {exc}]"

        gaps = _extract_list(analysis_text, ["gap", "missing", "unclear"])
        suggestions = _extract_list(analysis_text, ["suggest", "search", "query", "explore"])
        coverage = _extract_coverage(analysis_text)

        return {
            "coverage": coverage,
            "current_knowledge": analysis_text,
            "gaps": gaps,
            "suggestions": suggestions,
            "sources": sources,
            "communities_used": len(communities),
        }


def _extract_list(text: str, keywords: List[str]) -> List[str]:
    """Heuristic: extract bullet-point lines containing any of the keywords."""
    lines = text.split("\n")
    results = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "*", "•", "·")) or re.match(r"^\d+[\.\)]", line):
            if any(kw.lower() in line.lower() for kw in keywords):
                clean = re.sub(r"^[-*•·\d\.\)]+\s*", "", line)
                if clean:
                    results.append(clean)
    return results[:10]


def _extract_coverage(text: str) -> str:
    for level in ("high", "medium", "low"):
        if level in text.lower():
            return level
    return "unknown"


# ---------------------------------------------------------------------------
# EvalAgent — KB health
# ---------------------------------------------------------------------------

class EvalAgent:
    """Evaluates KB health: coverage, coherence, connectivity."""

    def __init__(self, client: OpenAI, model: str, kv: JsonKVStorage, graph: GraphStorage):
        self._client = client
        self._model = model
        self._kv = kv
        self._graph = graph

    def run(self) -> Dict[str, Any]:
        import networkx as nx

        g = self._graph._graph
        total_nodes = g.number_of_nodes()
        total_edges = g.number_of_edges()
        total_chunks = len(self._kv)

        entity_nodes = [n for n, d in g.nodes(data=True) if d.get("role") == "entity"]
        hyperedge_nodes = [n for n, d in g.nodes(data=True) if d.get("role") == "hyperedge"]
        asset_nodes = [n for n, d in g.nodes(data=True) if d.get("role") == "asset"]
        cross_modal_edges = [n for n, d in g.nodes(data=True)
                             if d.get("role") == "hyperedge" and d.get("kind") == "cross_modal"]

        isolated = [n for n in g.nodes() if g.degree(n) == 0]
        ent_degrees = [(n, g.degree(n)) for n in entity_nodes]
        ent_degrees.sort(key=lambda x: x[1], reverse=True)
        top_entities = [{"name": n, "degree": d} for n, d in ent_degrees[:10]]

        avg_degree = (sum(dict(g.degree()).values()) / total_nodes) if total_nodes else 0.0

        isolation_ratio = len(isolated) / max(1, total_nodes)
        health_score = max(0.0, 1.0 - isolation_ratio)
        if total_chunks == 0:
            health_score = 0.0

        summary = ""
        if total_nodes > 0:
            try:
                stats_text = (
                    f"Knowledge Base Statistics:\n"
                    f"- Total nodes: {total_nodes}\n"
                    f"- Entity nodes: {len(entity_nodes)}\n"
                    f"- Hyperedge nodes: {len(hyperedge_nodes)}\n"
                    f"- Cross-modal hyperedges: {len(cross_modal_edges)}\n"
                    f"- Asset nodes: {len(asset_nodes)}\n"
                    f"- Total edges: {total_edges}\n"
                    f"- Chunks: {total_chunks}\n"
                    f"- Isolated nodes: {len(isolated)}\n"
                    f"- Average degree: {avg_degree:.2f}\n"
                    f"- Top entities: {[e['name'] for e in top_entities[:5]]}\n"
                )
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=[{"role": "user", "content": (
                        f"{stats_text}\n\nWrite a 2-3 sentence health assessment of this typed "
                        "omnimodal knowledge hypergraph. Comment on coverage, connectivity, "
                        "cross-modal integration, and any concerns."
                    )}],
                    temperature=0.3, max_tokens=300,
                )
                summary = response.choices[0].message.content or ""
            except Exception as exc:
                summary = f"[Could not generate summary: {exc}]"
        else:
            summary = "The knowledge base is empty. Ingest some files to get started."

        return {
            "total_nodes": total_nodes,
            "entity_nodes": len(entity_nodes),
            "hyperedge_nodes": len(hyperedge_nodes),
            "asset_nodes": len(asset_nodes),
            "cross_modal_hyperedges": len(cross_modal_edges),
            "total_edges": total_edges,
            "total_chunks": total_chunks,
            "isolated_nodes": isolated[:20],
            "isolated_count": len(isolated),
            "top_entities": top_entities,
            "avg_degree": round(avg_degree, 3),
            "health_score": round(health_score, 3),
            "summary": summary,
        }
