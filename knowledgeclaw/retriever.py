"""
Subgraph-level retriever (C3 fine tier) for KnowledgeClaw.

Implements the *fine* half of the hierarchical multi-granular retrieval:

    query → embed
          ├─→ entity VDB → graph traversal → hyperedge sources → chunks
          └─→ hyperedge VDB → hyperedge sources → chunks
          → RRF fusion → top-k chunks

The community-level retrieval (coarse tier) and the intent router that
chooses between the two tiers live in `community.py` and `router.py`.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from knowledgeclaw.builder import GRAPH_FIELD_SEP
from knowledgeclaw.storage import GraphStorage, JsonKVStorage, VectorStorage

logger = logging.getLogger(__name__)

_RRF_K = 4  # standard RRF constant


def retrieve(
    query: str,
    embedder,
    kv: JsonKVStorage,
    vdb_entity: VectorStorage,
    vdb_hyperedge: VectorStorage,
    graph: GraphStorage,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Subgraph-level retrieval — returns top-k text chunk dicts."""
    if not query.strip():
        return []

    query_vec = embedder.embed_one(query)

    # --- Entity vector search ------------------------------------------------
    entity_hits: List[Dict] = []
    try:
        entity_hits = vdb_entity.search(query_vec, top_k=top_k * 2)
    except Exception as exc:
        logger.warning("Entity VDB search failed: %s", exc)

    entity_chunk_ids: List[str] = []
    for hit in entity_hits:
        ent_node = hit.get("entity_id") or hit.get("entity_name", "")
        if not ent_node:
            continue
        # 1) Walk back to hyperedges that mention this entity
        try:
            for he_node in graph.predecessors(ent_node):
                node_data = graph.get_node(he_node)
                if not node_data or node_data.get("role") != "hyperedge":
                    continue
                src_ids = node_data.get("source_id", "")
                for cid in src_ids.split(GRAPH_FIELD_SEP):
                    cid = cid.strip()
                    if cid and cid not in entity_chunk_ids:
                        entity_chunk_ids.append(cid)
        except Exception as exc:
            logger.debug("Graph traversal error for %s: %s", ent_node, exc)

        # 2) Direct entity source_ids
        node_data = graph.get_node(ent_node)
        if node_data:
            src_ids = node_data.get("source_id", "")
            for cid in src_ids.split(GRAPH_FIELD_SEP):
                cid = cid.strip()
                if cid and cid not in entity_chunk_ids:
                    entity_chunk_ids.append(cid)

    # --- Hyperedge vector search --------------------------------------------
    hyperedge_hits: List[Dict] = []
    try:
        hyperedge_hits = vdb_hyperedge.search(query_vec, top_k=top_k * 2)
    except Exception as exc:
        logger.warning("Hyperedge VDB search failed: %s", exc)

    hyperedge_chunk_ids: List[str] = []
    for hit in hyperedge_hits:
        he_node = hit.get("hyperedge_id") or hit.get("hyperedge_name", "")
        if not he_node:
            continue
        node_data = graph.get_node(he_node)
        if node_data:
            src_ids = node_data.get("source_id", "")
            for cid in src_ids.split(GRAPH_FIELD_SEP):
                cid = cid.strip()
                if cid and cid not in hyperedge_chunk_ids:
                    hyperedge_chunk_ids.append(cid)
        else:
            src_id = hit.get("source_id", "")
            if src_id and src_id not in hyperedge_chunk_ids:
                hyperedge_chunk_ids.append(src_id)

    # --- RRF fusion ----------------------------------------------------------
    rrf_scores: Dict[str, float] = {}
    for rank, cid in enumerate(entity_chunk_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + _RRF_K)
    for rank, cid in enumerate(hyperedge_chunk_ids):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rank + _RRF_K)

    if not rrf_scores:
        results: List[Dict[str, Any]] = []
        for hit in hyperedge_hits[:top_k]:
            src = hit.get("source_id", "")
            if src:
                chunk = kv.get(src)
                if chunk:
                    results.append({**chunk, "score": float(hit.get("__score__", 0)), "chunk_id": src})
        return results

    sorted_ids = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]
    results = []
    for cid in sorted_ids:
        chunk = kv.get(cid)
        if chunk:
            results.append({**chunk, "score": rrf_scores[cid], "chunk_id": cid})
    return results
