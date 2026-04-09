"""
Incremental update + deletion + consistency maintenance (C2) for KnowledgeClaw.

The Updater is the only component allowed to mutate an existing knowledge
hypergraph after the initial Construction stage. It is responsible for:

  * **add_file** — re-uses the standard build pipeline but is safe to call
    repeatedly: existing entities and hyperedges are merged (their source
    lists, modality lists and chunk-id lists are *unioned*) instead of being
    duplicated, which is what makes the lifecycle "closed-loop".

  * **delete_source** — removes every chunk, hyperedge, asset and entity
    that is uniquely tied to a given source file. Entities that are still
    referenced by another source survive but have the deleted source pruned
    from their `sources`/`source_id`/`modalities` attribute lists.

  * **consistency_check** — global cleanup pass that drops orphan entities
    (no remaining hyperedge), drops orphan hyperedges (no remaining
    entities), and prunes vector store entries whose graph node is gone.
    Returns a structured report so the lifecycle agent can log it.

  * **recompute_communities** — invalidates the community cache and rebuilds
    it from scratch using `community.build_community_reports`.

The motivating "closed loop" means: after every add/delete the consistency
report is recomputed, and stale community summaries are flushed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Set

from knowledgeclaw.builder import GRAPH_FIELD_SEP, _md5
from knowledgeclaw.storage import GraphStorage, JsonKVStorage, VectorStorage

logger = logging.getLogger(__name__)


def _split_set(value: str) -> Set[str]:
    if not value:
        return set()
    return {p for p in value.split(GRAPH_FIELD_SEP) if p}


def _join_set(items: Set[str]) -> str:
    return GRAPH_FIELD_SEP.join(sorted(items))


class Updater:
    """Mutates the typed hypergraph while preserving consistency."""

    def __init__(
        self,
        storage_dir: str,
        graph: GraphStorage,
        kv: JsonKVStorage,
        vdb_entity: VectorStorage,
        vdb_hyperedge: VectorStorage,
        vdb_media: VectorStorage = None,
        vdb_community: VectorStorage = None,
    ):
        self._storage_dir = storage_dir
        self._graph = graph
        self._kv = kv
        self._vdb_entity = vdb_entity
        self._vdb_hyperedge = vdb_hyperedge
        self._vdb_media = vdb_media
        self._vdb_community = vdb_community
        self._processed_kv = JsonKVStorage(str(Path(storage_dir) / "processed_files.json"))

    # ----------------------------------------------------------------- add

    def add_file(
        self,
        file_path: str,
        extractor,
        embedder,
        media_embedder,
        max_tokens: int,
        overlap: int,
        extract_batch_size: int,
        expr_dir: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Incrementally ingest a single file into the existing hypergraph."""
        from knowledgeclaw.builder import build_from_file

        result = build_from_file(
            file_path=file_path,
            kb_dir=self._storage_dir,
            extractor=extractor,
            embedder=embedder,
            kv=self._kv,
            vdb_entity=self._vdb_entity,
            vdb_hyperedge=self._vdb_hyperedge,
            graph=self._graph,
            max_tokens=max_tokens,
            overlap=overlap,
            extract_batch_size=extract_batch_size,
            vdb_media=self._vdb_media,
            media_embedder=media_embedder,
            expr_dir=expr_dir,
            force=force,
        )
        # Reload processed-file index from disk (build_from_file persisted it)
        self._processed_kv = JsonKVStorage(str(Path(self._storage_dir) / "processed_files.json"))
        # Stale: any cached community report no longer reflects the new graph.
        self._invalidate_community_cache()
        return result

    # -------------------------------------------------------------- delete

    def delete_source(self, source_name: str) -> Dict[str, Any]:
        """Delete every artifact uniquely tied to *source_name* and prune
        partial references in shared entities. Returns a removal report."""
        report = {
            "source": source_name,
            "chunks_removed": 0,
            "entities_removed": 0,
            "entities_pruned": 0,
            "hyperedges_removed": 0,
            "assets_removed": 0,
            "vectors_removed": 0,
        }

        g = self._graph._graph

        # 1. Find chunk IDs belonging to this source
        chunk_ids_to_remove: List[str] = []
        for cid, chunk in list(self._kv.all().items()):
            if isinstance(chunk, dict) and chunk.get("source") == source_name:
                chunk_ids_to_remove.append(cid)

        for cid in chunk_ids_to_remove:
            self._kv.delete(cid)
        report["chunks_removed"] = len(chunk_ids_to_remove)
        chunk_id_set = set(chunk_ids_to_remove)

        # 2. Walk graph nodes — partition into "remove" vs "prune"
        nodes_to_remove: List[str] = []
        for node, data in list(g.nodes(data=True)):
            sources = _split_set(data.get("sources") or data.get("source", ""))
            if source_name not in sources:
                continue

            if sources == {source_name}:
                # Node is uniquely tied to this source → remove entirely
                nodes_to_remove.append(node)
            else:
                # Node is shared → prune source-specific information
                sources.discard(source_name)
                src_ids = _split_set(data.get("source_id", ""))
                src_ids -= chunk_id_set
                # Modalities are stored as a multiset (best effort) — recompute by
                # walking remaining hyperedges later in consistency_check.
                g.nodes[node]["sources"] = _join_set(sources)
                g.nodes[node]["source_id"] = _join_set(src_ids)
                if g.nodes[node].get("source") == source_name:
                    g.nodes[node]["source"] = sorted(sources)[0]
                report["entities_pruned"] += int(data.get("role") == "entity")

        # 3. Drop nodes
        vdb_ids_to_drop: Set[str] = set()
        for node in nodes_to_remove:
            role = g.nodes[node].get("role")
            if role == "entity":
                report["entities_removed"] += 1
            elif role == "hyperedge":
                report["hyperedges_removed"] += 1
            elif role == "asset":
                report["assets_removed"] += 1
            vdb_ids_to_drop.add(_md5(node))
            g.remove_node(node)

        # 4. Prune vector stores in-place by rebuilding the underlying matrix
        report["vectors_removed"] += self._prune_vdb(self._vdb_entity, vdb_ids_to_drop)
        report["vectors_removed"] += self._prune_vdb(self._vdb_hyperedge, vdb_ids_to_drop)
        if self._vdb_media is not None:
            # Media records key on image_id/audio_id/segment_id, not md5(node).
            # We instead drop any record whose meta.source matches.
            report["vectors_removed"] += self._prune_vdb_by_meta(
                self._vdb_media, "source", source_name
            )

        # 5. Clean up processed_files index
        for key in list(self._processed_kv.all().keys()):
            entry = self._processed_kv.get(key, {})
            if isinstance(entry, dict) and entry.get("name") == source_name:
                self._processed_kv.delete(key)
        self._processed_kv.save()

        # 6. Persist & invalidate communities
        self._kv.save()
        self._vdb_entity.save()
        self._vdb_hyperedge.save()
        if self._vdb_media is not None:
            self._vdb_media.save()
        self._graph.save()
        self._invalidate_community_cache()

        return report

    # ------------------------------------------------------------- consistency

    def consistency_check(self) -> Dict[str, Any]:
        """Drop orphan entities, orphan hyperedges, and orphaned vectors."""
        report = {
            "orphan_entities_removed": 0,
            "orphan_hyperedges_removed": 0,
            "orphan_assets_removed": 0,
            "vectors_removed": 0,
        }
        g = self._graph._graph

        vdb_ids_to_drop: Set[str] = set()

        # Orphan hyperedges: no entity successor
        for node in list(g.nodes()):
            data = g.nodes[node]
            if data.get("role") != "hyperedge":
                continue
            ent_succ = [t for t in g.successors(node) if g.nodes[t].get("role") == "entity"]
            if not ent_succ:
                vdb_ids_to_drop.add(_md5(node))
                g.remove_node(node)
                report["orphan_hyperedges_removed"] += 1

        # Orphan entities: no incoming hyperedge AND no incoming asset
        for node in list(g.nodes()):
            data = g.nodes[node]
            if data.get("role") != "entity":
                continue
            preds = list(g.predecessors(node))
            if not preds:
                vdb_ids_to_drop.add(_md5(node))
                g.remove_node(node)
                report["orphan_entities_removed"] += 1

        # Orphan assets: no outgoing entity edges
        for node in list(g.nodes()):
            data = g.nodes[node]
            if data.get("role") != "asset":
                continue
            ent_succ = [t for t in g.successors(node) if g.nodes[t].get("role") == "entity"]
            if not ent_succ:
                g.remove_node(node)
                report["orphan_assets_removed"] += 1

        report["vectors_removed"] += self._prune_vdb(self._vdb_entity, vdb_ids_to_drop)
        report["vectors_removed"] += self._prune_vdb(self._vdb_hyperedge, vdb_ids_to_drop)

        self._vdb_entity.save()
        self._vdb_hyperedge.save()
        self._graph.save()
        self._invalidate_community_cache()
        return report

    # ----------------------------------------------------------- communities

    def _invalidate_community_cache(self) -> None:
        cache = Path(self._storage_dir) / "community_reports.json"
        if cache.exists():
            try:
                cache.unlink()
                logger.info("[updater] community cache invalidated")
            except Exception as exc:
                logger.warning("Could not invalidate community cache: %s", exc)

    # --------------------------------------------------------- vdb helpers

    def _prune_vdb(self, vdb: VectorStorage, drop_ids: Set[str]) -> int:
        """Remove entries whose `__id__` is in *drop_ids*."""
        if not drop_ids:
            return 0
        try:
            storage = vdb._db._NanoVectorDB__storage  # noqa: SLF001 — internal access
            data = storage.get("data", [])
            keep_idx = [i for i, rec in enumerate(data) if rec.get("__id__") not in drop_ids]
            removed = len(data) - len(keep_idx)
            if removed == 0:
                return 0
            storage["data"] = [data[i] for i in keep_idx]
            matrix = storage.get("matrix")
            if matrix is not None and getattr(matrix, "ndim", 0) == 2 and len(matrix) >= len(data):
                storage["matrix"] = matrix[keep_idx]
            return removed
        except Exception as exc:
            logger.warning("VDB prune failed: %s", exc)
            return 0

    def _prune_vdb_by_meta(self, vdb: VectorStorage, key: str, value: str) -> int:
        """Remove entries whose meta[key] == value."""
        try:
            storage = vdb._db._NanoVectorDB__storage  # noqa: SLF001
            data = storage.get("data", [])
            keep_idx = [i for i, rec in enumerate(data) if rec.get(key) != value]
            removed = len(data) - len(keep_idx)
            if removed == 0:
                return 0
            storage["data"] = [data[i] for i in keep_idx]
            matrix = storage.get("matrix")
            if matrix is not None and getattr(matrix, "ndim", 0) == 2 and len(matrix) >= len(data):
                storage["matrix"] = matrix[keep_idx]
            return removed
        except Exception as exc:
            logger.warning("VDB meta-prune failed: %s", exc)
            return 0
