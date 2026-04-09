"""
Community-level retrieval (C3 coarse tier) for KnowledgeClaw.

Implements the *coarse* half of the hierarchical multi-granular retrieval:

  1. Run hierarchical community detection over the entity-induced subgraph
     of the typed hypergraph (Louvain → fall back to greedy modularity →
     fall back to connected components).
  2. For each community, build a textual *community report* by sampling
     entities + their hyperedge sentences and asking the LLM to summarise
     the theme.
  3. Embed the community reports and store them in a dedicated community
     vector store. At query time we search this VDB to dispatch open-ended
     synthesis queries to the right community.

Community reports are persisted alongside the rest of the storage so that
subsequent processes can reuse them without paying the LLM cost again.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from knowledgeclaw.builder import GRAPH_FIELD_SEP, _md5
from knowledgeclaw.storage import GraphStorage, JsonKVStorage, VectorStorage

logger = logging.getLogger(__name__)


COMMUNITY_REPORT_PROMPT = """You are summarising one community of entities from a personal knowledge hypergraph.

Community entities (canonical IDs): {entities}

Sample propositions / hyperedges that mention these entities:
{propositions}

Modalities present in this community: {modalities}

Write a concise community report with the following structure (no markdown headers, just plain prose):
1. THEME — a single-sentence label for what this community is about.
2. KEY ENTITIES — list the 3-5 most central entities and their role.
3. SUMMARY — a 3-5 sentence synthesis of the relationships and facts.
4. CROSS-MODAL NOTES — if multiple modalities appear, note how they reinforce each other.

Keep the whole report under 200 words and avoid generic boilerplate."""


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_communities(graph: GraphStorage, max_communities: int = 64) -> List[List[str]]:
    """Run hierarchical community detection on the entity-induced subgraph.

    Returns a list of communities (each a list of entity node IDs)."""
    g = graph._graph
    entity_nodes = [n for n, d in g.nodes(data=True) if d.get("role") == "entity"]
    if not entity_nodes:
        return []

    # Build an undirected co-occurrence graph: two entities are connected if
    # they share at least one hyperedge.
    co_g = nx.Graph()
    co_g.add_nodes_from(entity_nodes)
    he_nodes = [n for n, d in g.nodes(data=True) if d.get("role") == "hyperedge"]
    for he in he_nodes:
        members = [t for t in g.successors(he) if g.nodes[t].get("role") == "entity"]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if co_g.has_edge(a, b):
                    co_g[a][b]["weight"] += 1
                else:
                    co_g.add_edge(a, b, weight=1)

    if co_g.number_of_edges() == 0:
        # Each isolated entity is its own community
        return [[n] for n in entity_nodes][:max_communities]

    communities: List[Set[str]] = []
    try:
        from networkx.algorithms.community import louvain_communities
        communities = louvain_communities(co_g, weight="weight", seed=0)
        logger.info("[community] Louvain detected %d communities", len(communities))
    except Exception as exc:
        logger.warning("[community] Louvain unavailable (%s); using greedy modularity", exc)
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(co_g, weight="weight"))
        except Exception as exc2:
            logger.warning("[community] Greedy modularity failed (%s); using CC", exc2)
            communities = [set(c) for c in nx.connected_components(co_g)]

    # Sort communities by size (desc) and cap
    communities = sorted(communities, key=lambda c: -len(c))[:max_communities]
    return [list(c) for c in communities]


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------

def _gather_community_propositions(
    graph: GraphStorage, members: List[str], max_props: int = 12
) -> Dict[str, Any]:
    g = graph._graph
    seen: Set[str] = set()
    propositions: List[str] = []
    modalities: Set[str] = set()
    for ent in members:
        for he in g.predecessors(ent):
            if he in seen:
                continue
            data = g.nodes[he]
            if data.get("role") != "hyperedge":
                continue
            seen.add(he)
            sent = data.get("sentence") or he
            propositions.append(f"- {sent}")
            mods = data.get("modalities") or data.get("modality", "")
            for m in str(mods).split(GRAPH_FIELD_SEP):
                if m:
                    modalities.add(m)
            if len(propositions) >= max_props:
                break
        if len(propositions) >= max_props:
            break
    return {"propositions": propositions, "modalities": sorted(modalities)}


def build_community_reports(
    graph: GraphStorage,
    llm_client,
    model: str,
    max_communities: int = 64,
    min_size: int = 2,
) -> List[Dict[str, Any]]:
    """Detect communities and ask the LLM to summarise each one."""
    communities = detect_communities(graph, max_communities=max_communities)
    reports: List[Dict[str, Any]] = []
    for idx, members in enumerate(communities):
        if len(members) < min_size:
            continue
        info = _gather_community_propositions(graph, members)
        ent_sample = members[:10]
        prompt = COMMUNITY_REPORT_PROMPT.format(
            entities=", ".join(ent_sample),
            propositions="\n".join(info["propositions"]) or "(no hyperedges)",
            modalities=", ".join(info["modalities"]) or "(none)",
        )
        try:
            resp = llm_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400,
            )
            report = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning("Community %d report failed: %s", idx, exc)
            report = (
                f"Community of {len(members)} entities including "
                f"{', '.join(ent_sample[:3])}. Modalities: {', '.join(info['modalities']) or 'text'}."
            )

        reports.append({
            "community_id": f"comm_{idx:04d}",
            "size": len(members),
            "members": members,
            "modalities": info["modalities"],
            "report": report,
        })
    return reports


def persist_community_reports(reports: List[Dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(reports, f, ensure_ascii=False, indent=2)


def load_community_reports(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Could not decode community reports at %s", path)
        return []


def embed_community_reports(
    reports: List[Dict[str, Any]],
    embedder,
    vdb_community: VectorStorage,
) -> int:
    if not reports:
        return 0
    texts = [r["report"] or r["community_id"] for r in reports]
    vectors = embedder.embed(texts)
    entries = []
    for r, vec in zip(reports, vectors):
        entries.append({
            "id": r["community_id"],
            "vector": vec,
            "meta": {
                "content": r["report"],
                "community_id": r["community_id"],
                "size": r["size"],
                "modalities": ",".join(r.get("modalities", [])),
            },
        })
    vdb_community.upsert_batch(entries)
    vdb_community.save()
    return len(entries)


# ---------------------------------------------------------------------------
# Public retrieval API
# ---------------------------------------------------------------------------

def community_search(
    query: str,
    embedder,
    vdb_community: VectorStorage,
    reports: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Return top-k community reports for a query."""
    if not query.strip() or len(vdb_community) == 0:
        return []
    q_vec = embedder.embed_one(query)
    hits = vdb_community.search(q_vec, top_k=top_k)
    by_id = {r["community_id"]: r for r in reports}
    results: List[Dict[str, Any]] = []
    for h in hits:
        cid = h.get("community_id") or h.get("__id__")
        if not cid:
            continue
        meta = by_id.get(cid, {})
        results.append({
            "community_id": cid,
            "score": float(h.get("__score__", 0.0)),
            "report": h.get("content") or meta.get("report", ""),
            "size": meta.get("size", 0),
            "modalities": meta.get("modalities", []),
            "members": meta.get("members", [])[:10],
        })
    return results
