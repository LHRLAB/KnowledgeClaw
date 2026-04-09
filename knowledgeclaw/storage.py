"""
Storage backends for KnowledgeClaw.

- JsonKVStorage: JSON-file-backed key-value store
- VectorStorage: wraps nano_vectordb.NanoVectorDB
- GraphStorage: wraps networkx.Graph (saved as GraphML)
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JsonKVStorage
# ---------------------------------------------------------------------------

class JsonKVStorage:
    """Persistent JSON key-value store backed by a single .json file."""

    def __init__(self, file_path: str):
        self._path = Path(file_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                self._data: Dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("Corrupt KV file %s – starting fresh.", self._path)
                self._data = {}
        else:
            self._data = {}

    # --- read -----------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def all(self) -> Dict[str, Any]:
        return dict(self._data)

    def filter_new(self, keys: List[str]) -> List[str]:
        """Return only keys that are NOT yet stored."""
        return [k for k in keys if k not in self._data]

    # --- write ----------------------------------------------------------------

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def upsert(self, items: Dict[str, Any]) -> None:
        """Update multiple key-value pairs and persist."""
        self._data.update(items)
        self.save()

    def delete(self, key: str) -> None:
        self._data.pop(key, None)

    def save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: str) -> bool:
        return key in self._data


# ---------------------------------------------------------------------------
# VectorStorage
# ---------------------------------------------------------------------------

class VectorStorage:
    """
    Thin wrapper around nano_vectordb.NanoVectorDB.

    nano_vectordb stores: a numpy array of embeddings + a list of metadata dicts.
    Each metadata dict MUST have '__id__' and may have arbitrary extra fields.
    """

    def __init__(self, storage_file: str, embedding_dim: int):
        from nano_vectordb import NanoVectorDB  # imported lazily so tests can mock

        self._path = Path(storage_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._dim = embedding_dim
        self._db = NanoVectorDB(embedding_dim, storage_file=str(self._path))
        # nano_vectordb initialises matrix as np.array([]) (1-D) for a fresh DB.
        # This causes an IndexError when upsert tries to update existing rows via
        # matrix[i] because the 1-D empty array has size 0.  Ensure it is always
        # a proper 2-D array of shape (N, dim) so row-index access works correctly.
        _s = self._db._NanoVectorDB__storage
        if _s["matrix"].ndim < 2:
            _s["matrix"] = np.zeros((len(_s.get("data", [])), embedding_dim), dtype=np.float32)

    def upsert(self, id: str, vector: List[float], meta: Dict[str, Any]) -> None:
        """Insert or update a single vector entry."""
        data = {"__id__": id, **meta}
        self._db.upsert([{"__id__": id, "__vector__": np.array(vector, dtype=np.float32), **meta}])

    def upsert_batch(self, entries: List[Dict[str, Any]]) -> None:
        """
        Bulk upsert.

        Each entry: {"id": str, "vector": list[float], "meta": dict}
        """
        records = []
        for e in entries:
            rec = {"__id__": e["id"], "__vector__": np.array(e["vector"], dtype=np.float32)}
            rec.update(e.get("meta", {}))
            records.append(rec)
        if records:
            self._db.upsert(records)

    def search(self, query_vec: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Return top_k nearest neighbours.

        Result: list of dicts with all stored metadata fields + '__score__' and '__id__'.
        """
        q = np.array(query_vec, dtype=np.float32)
        results = self._db.query(q, top_k=top_k)
        return results  # nano_vectordb returns list of dicts already

    def save(self) -> None:
        self._db.save()

    def __len__(self) -> int:
        try:
            return len(self._db._NanoVectorDB__storage["data"])
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# GraphStorage
# ---------------------------------------------------------------------------

class GraphStorage:
    """
    Wraps a NetworkX directed graph, persisted as GraphML.

    Node conventions:
      - Entity node:    id = "ENTITY_NAME | Source: filename"
                        attrs: {role="entity", source_id="chunk1<SEP>chunk2"}
      - Hyperedge node: id = "<hyperedge>proposition | Source: filename"
                        attrs: {role="hyperedge", source_id="chunk_id"}

    Edge direction: hyperedge_node -> entity_node
    """

    def __init__(self, storage_file: str):
        self._path = Path(storage_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            try:
                self._graph: nx.DiGraph = nx.read_graphml(str(self._path))
                logger.info("Loaded graph from %s (%d nodes, %d edges)",
                            self._path, self._graph.number_of_nodes(), self._graph.number_of_edges())
            except Exception as exc:
                logger.warning("Could not load graph %s: %s – starting fresh.", self._path, exc)
                self._graph = nx.DiGraph()
        else:
            self._graph = nx.DiGraph()

    # --- node operations ------------------------------------------------------

    def upsert_node(self, node_id: str, attrs: Dict[str, Any]) -> None:
        """
        Insert or update a node.

        If the node exists and has a 'source_id' field, append new source ids
        using GRAPH_FIELD_SEP rather than overwriting.
        """
        from knowledgeclaw.builder import GRAPH_FIELD_SEP  # avoid circular at module level

        if self._graph.has_node(node_id):
            existing = dict(self._graph.nodes[node_id])
            if "source_id" in attrs and "source_id" in existing:
                existing_ids = set(existing["source_id"].split(GRAPH_FIELD_SEP))
                new_ids = set(attrs["source_id"].split(GRAPH_FIELD_SEP))
                merged = existing_ids | new_ids
                attrs = {**attrs, "source_id": GRAPH_FIELD_SEP.join(sorted(merged))}
            self._graph.nodes[node_id].update(attrs)
        else:
            self._graph.add_node(node_id, **attrs)

    def upsert_edge(self, src: str, dst: str, attrs: Dict[str, Any] | None = None) -> None:
        self._graph.add_edge(src, dst, **(attrs or {}))

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if self._graph.has_node(node_id):
            return dict(self._graph.nodes[node_id])
        return None

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def get_edges(self, node_id: str) -> List[tuple]:
        """Return list of (src, dst, attrs) for all edges connected to node_id."""
        edges = []
        for src, dst, data in self._graph.out_edges(node_id, data=True):
            edges.append((src, dst, data))
        for src, dst, data in self._graph.in_edges(node_id, data=True):
            edges.append((src, dst, data))
        return edges

    def neighbors(self, node_id: str) -> List[str]:
        """Return successor node IDs (out-edges)."""
        return list(self._graph.successors(node_id))

    def predecessors(self, node_id: str) -> List[str]:
        """Return predecessor node IDs (in-edges)."""
        return list(self._graph.predecessors(node_id))

    def nodes_by_role(self, role: str) -> List[str]:
        return [n for n, d in self._graph.nodes(data=True) if d.get("role") == role]

    def all_nodes(self) -> List[tuple]:
        """Return list of (node_id, attrs)."""
        return [(n, dict(d)) for n, d in self._graph.nodes(data=True)]

    def all_edges(self) -> List[tuple]:
        return [(s, t, dict(d)) for s, t, d in self._graph.edges(data=True)]

    # --- persistence ----------------------------------------------------------

    def save(self) -> None:
        nx.write_graphml(self._graph, str(self._path))
        logger.debug("Graph saved to %s", self._path)

    # --- stats ----------------------------------------------------------------

    def number_of_nodes(self) -> int:
        return self._graph.number_of_nodes()

    def number_of_edges(self) -> int:
        return self._graph.number_of_edges()

    def degree(self, node_id: str) -> int:
        return self._graph.degree(node_id)
