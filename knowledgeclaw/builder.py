"""
Typed Omnimodal Hypergraph builder for KnowledgeClaw.

Implements C1 (Omnimodal typed hypergraph) from PAPER.md:
  * Canonical entity nodes (one node per entity, regardless of source) so that
    the same entity mentioned in multiple files / modalities is represented
    by a single graph node — required for the C2 lifecycle consistency story.
  * Typed hyperedges with explicit `modality` ∈ {text, code, json, csv, pdf,
    image, audio, video, cross_modal}.
  * Asset nodes for non-text artifacts (PDF page images, audio segments,
    video clips, standalone images) so that media files become first-class
    citizens of the hypergraph.
  * Cross-modal hyperedges that connect an entity's modality-specific
    occurrences when it appears in chunks of more than one modality within a
    single source — these are the edges that prior text-centric KG-RAG cannot
    express (L1 modality gap).

Pipeline (per file):

   raw file → ingest_file()  ─┬─→  text chunks  ─→  LLM proposition extractor
                              │                            │
                              │                            ▼
                              │                    typed hyperedges
                              │                    + canonical entities
                              │
                              └─→  media records  ─→  asset nodes
                                                       + entity ↔ asset edges
                                                       + cross-modal hyperedges
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from knowledgeclaw.ingestor import ingest_file, is_supported
from knowledgeclaw.storage import GraphStorage, JsonKVStorage, VectorStorage

logger = logging.getLogger(__name__)

# Field separator used inside <SEP>-joined attribute values (compatible with
# the GraphML serialiser — must not contain XML special chars).
GRAPH_FIELD_SEP = "<SEP>"

# Modalities recognised by the typed hypergraph.
MODALITIES = ("text", "code", "json", "csv", "pdf", "image", "audio", "video", "cross_modal")

# Hidden / generated directories to skip when walking the KB folder.
_SKIP_DIRS = {
    ".knowledgeclaw", ".openclaw", ".git", "__pycache__",
    "node_modules", ".venv", "venv", ".idea", ".vscode", "expr",
}


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _chunk_id(source: str, chunk_index: int) -> str:
    return f"{_md5(source)}_{chunk_index}"


def canonical_entity_id(entity_name: str) -> str:
    """Canonical entity node ID — one node per (case-folded) entity name."""
    return entity_name.strip().upper()


def canonical_hyperedge_id(sentence: str) -> str:
    """Stable hyperedge node ID derived from the proposition text."""
    return f"<HE>{_md5(sentence.strip().lower())}"


def asset_node_id(modality: str, path: str) -> str:
    """Asset (media) node ID — keyed by modality + filename."""
    return f"<ASSET:{modality}>{Path(path).name}"


def cross_modal_hyperedge_id(source: str, entity: str, modalities: List[str]) -> str:
    key = f"{source}|{entity}|{'+'.join(sorted(set(modalities)))}"
    return f"<XM>{_md5(key)}"


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def build_from_folder(
    folder_path: str,
    kb_dir: str,
    extractor,
    embedder,
    kv: JsonKVStorage,
    vdb_entity: VectorStorage,
    vdb_hyperedge: VectorStorage,
    graph: GraphStorage,
    max_tokens: int = 512,
    overlap: int = 64,
    extract_batch_size: int = 5,
    vdb_media: Optional[VectorStorage] = None,
    media_embedder=None,
    expr_dir: str = "./expr",
) -> Dict[str, Any]:
    """Walk a folder and ingest every supported file into the typed hypergraph."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    processed_kv = JsonKVStorage(str(Path(kb_dir) / "processed_files.json"))
    processed_set: Set[str] = set(processed_kv.all().keys())

    stats = {
        "files_found": 0,
        "files_skipped": 0,
        "files_processed": 0,
        "chunks_created": 0,
        "entities_new": 0,
        "hyperedges_new": 0,
        "cross_modal_edges": 0,
        "asset_nodes": 0,
        "media_records": 0,
        "errors": 0,
    }

    all_files: List[Path] = []
    for root, dirs, files in os.walk(str(folder)):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for fname in files:
            fpath = Path(root) / fname
            if is_supported(str(fpath)):
                all_files.append(fpath)

    stats["files_found"] = len(all_files)

    # Filter out already-processed files
    to_process = []
    for fpath in all_files:
        file_key = str(fpath.resolve())
        if file_key in processed_set:
            stats["files_skipped"] += 1
        else:
            to_process.append((fpath, file_key))

    # Parallel ingestion with lock-protected storage writes
    _storage_lock = threading.Lock()
    max_workers = int(os.environ.get("KC_INGEST_WORKERS", "40"))
    done_count = [0]

    def _do_file(fpath_and_key):
        fpath, file_key = fpath_and_key
        try:
            file_stats = _process_file(
                file_path=str(fpath),
                kv=kv,
                vdb_entity=vdb_entity,
                vdb_hyperedge=vdb_hyperedge,
                graph=graph,
                extractor=extractor,
                embedder=embedder,
                source_name=fpath.name,
                max_tokens=max_tokens,
                overlap=overlap,
                extract_batch_size=extract_batch_size,
                vdb_media=vdb_media,
                media_embedder=media_embedder,
                expr_dir=expr_dir,
                _lock=_storage_lock,
            )
            with _storage_lock:
                stats["files_processed"] += 1
                stats["chunks_created"] += file_stats["chunks"]
                stats["entities_new"] += file_stats["entities"]
                stats["hyperedges_new"] += file_stats["hyperedges"]
                stats["cross_modal_edges"] += file_stats["cross_modal_edges"]
                stats["asset_nodes"] += file_stats["asset_nodes"]
                stats["media_records"] += file_stats.get("media_records", 0)
                processed_kv.upsert({file_key: {
                    "name": fpath.name,
                    "chunks": file_stats["chunks"],
                    "modalities": list(file_stats["modalities"]),
                }})
                processed_set.add(file_key)
                done_count[0] += 1
                if done_count[0] % 50 == 0:
                    logger.info("Ingestion progress: %d/%d files", done_count[0], len(to_process))
        except Exception as exc:
            logger.error("Error processing %s: %s", fpath, exc)
            with _storage_lock:
                stats["errors"] += 1

    if to_process:
        logger.info("Ingesting %d files with %d workers...", len(to_process), max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            list(ex.map(_do_file, to_process))

    kv.save()
    vdb_entity.save()
    vdb_hyperedge.save()
    graph.save()
    if vdb_media is not None:
        vdb_media.save()

    return stats


def build_from_file(
    file_path: str,
    kb_dir: str,
    extractor,
    embedder,
    kv: JsonKVStorage,
    vdb_entity: VectorStorage,
    vdb_hyperedge: VectorStorage,
    graph: GraphStorage,
    max_tokens: int = 512,
    overlap: int = 64,
    extract_batch_size: int = 5,
    vdb_media: Optional[VectorStorage] = None,
    media_embedder=None,
    expr_dir: str = "./expr",
    force: bool = False,
) -> Dict[str, Any]:
    """Ingest a single file (used by CLI / Updater)."""
    fpath = Path(file_path)
    processed_kv = JsonKVStorage(str(Path(kb_dir) / "processed_files.json"))
    processed_set: Set[str] = set(processed_kv.all().keys())

    file_key = str(fpath.resolve())
    if file_key in processed_set and not force:
        return {
            "files_processed": 0, "chunks_created": 0, "entities_new": 0,
            "hyperedges_new": 0, "media_records": 0, "cross_modal_edges": 0,
            "asset_nodes": 0, "skipped": True,
        }

    file_stats = _process_file(
        file_path=str(fpath),
        kv=kv,
        vdb_entity=vdb_entity,
        vdb_hyperedge=vdb_hyperedge,
        graph=graph,
        extractor=extractor,
        embedder=embedder,
        source_name=fpath.name,
        max_tokens=max_tokens,
        overlap=overlap,
        extract_batch_size=extract_batch_size,
        vdb_media=vdb_media,
        media_embedder=media_embedder,
        expr_dir=expr_dir,
    )

    processed_kv.upsert({file_key: {
        "name": fpath.name,
        "chunks": file_stats["chunks"],
        "modalities": list(file_stats["modalities"]),
    }})
    kv.save()
    vdb_entity.save()
    vdb_hyperedge.save()
    graph.save()
    if vdb_media is not None:
        vdb_media.save()

    return {
        "files_processed": 1,
        "chunks_created": file_stats["chunks"],
        "entities_new": file_stats["entities"],
        "hyperedges_new": file_stats["hyperedges"],
        "cross_modal_edges": file_stats["cross_modal_edges"],
        "asset_nodes": file_stats["asset_nodes"],
        "media_records": file_stats.get("media_records", 0),
    }


# ---------------------------------------------------------------------------
# Internal pipeline
# ---------------------------------------------------------------------------

def _process_file(
    file_path: str,
    kv: JsonKVStorage,
    vdb_entity: VectorStorage,
    vdb_hyperedge: VectorStorage,
    graph: GraphStorage,
    extractor,
    embedder,
    source_name: str,
    max_tokens: int,
    overlap: int,
    extract_batch_size: int,
    vdb_media: Optional[VectorStorage] = None,
    media_embedder=None,
    expr_dir: str = "./expr",
    _lock: Optional[threading.Lock] = None,
) -> Dict[str, Any]:
    """Full processing pipeline for a single file.

    When _lock is provided, all storage writes are protected for thread safety.
    The LLM/embedding calls (Steps 1, 3, 5) run without the lock for parallelism.
    """

    # Step 1: ingest → (chunks, media_records) — NO lock needed (pure I/O + parsing)
    chunks, media_records = ingest_file(
        file_path, max_tokens=max_tokens, overlap=overlap, expr_dir=expr_dir
    )

    if not chunks and not media_records:
        return {
            "chunks": 0, "entities": 0, "hyperedges": 0,
            "media_records": 0, "cross_modal_edges": 0, "asset_nodes": 0,
            "modalities": set(),
        }

    # Assign canonical chunk_ids (local computation, no lock needed)
    file_modalities: Set[str] = set()
    for chunk in chunks:
        cid = chunk.get("chunk_id") or _chunk_id(source_name, chunk["chunk_index"])
        chunk["chunk_id"] = cid
        chunk["modality"] = chunk.get("type", "text")
        file_modalities.add(chunk["modality"])

    # Step 3: extract propositions (LLM calls — NO lock, this is the bottleneck)
    all_propositions: List[Dict] = []
    if chunks:
        chunk_texts = [c["content"] for c in chunks]
        for batch_start in range(0, len(chunk_texts), extract_batch_size):
            batch_texts = chunk_texts[batch_start: batch_start + extract_batch_size]
            batch_chunks = chunks[batch_start: batch_start + extract_batch_size]
            extraction_results = extractor.extract_batch(batch_texts)
            for chunk, propositions in zip(batch_chunks, extraction_results):
                for prop in propositions:
                    all_propositions.append({
                        "sentence": prop["sentence"],
                        "entities": prop["entities"],
                        "chunk_id": chunk["chunk_id"],
                        "source": source_name,
                        "modality": chunk["modality"],
                    })

    # Steps 2+4+5+6+7+8: all storage writes — LOCKED for thread safety
    _ctx = _lock if _lock else threading.Lock()  # dummy lock if none provided
    with _ctx:
        # Step 2: persist chunks
        for chunk in chunks:
            kv.set(chunk["chunk_id"], chunk)

    # Steps 4-7: graph + VDB writes — all under lock for thread safety
    entity_to_modalities: Dict[str, Set[str]] = defaultdict(set)
    entity_to_chunks: Dict[str, Set[str]] = defaultdict(set)
    new_entity_texts: List[str] = []
    new_entity_ids: List[Tuple[str, str, str, str]] = []
    new_hyperedge_texts: List[str] = []
    new_hyperedge_ids: List[Tuple[str, str, str]] = []

    with _ctx:
        # Step 4: build graph nodes
        for prop in all_propositions:
            sentence = prop["sentence"]
            chunk_id = prop["chunk_id"]
            source = prop["source"]
            modality = prop["modality"]

            he_node_id = canonical_hyperedge_id(sentence)
            graph.upsert_node(he_node_id, {
                "role": "hyperedge",
                "kind": "fact",
                "modality": modality,
                "source_id": chunk_id,
                "source": source,
                "sources": source,
                "modalities": modality,
                "sentence": sentence,
            })

            he_vdb_id = _md5(he_node_id)
            new_hyperedge_texts.append(sentence)
            new_hyperedge_ids.append((he_vdb_id, he_node_id, source))

            for entity in prop["entities"]:
                if isinstance(entity, dict):
                    entity_name = entity.get("name", "").strip()
                    entity_name_zh = entity.get("name_zh", entity_name).strip()
                else:
                    entity_name = str(entity).strip()
                    entity_name_zh = entity_name
                if not entity_name:
                    continue
                ent_node_id = canonical_entity_id(entity_name)
                graph.upsert_node(ent_node_id, {
                    "role": "entity",
                    "name": entity_name,
                    "name_zh": entity_name_zh,
                    "source_id": chunk_id,
                    "source": source,
                    "sources": source,
                    "modalities": modality,
                })
                graph.upsert_edge(he_node_id, ent_node_id)

                entity_to_modalities[ent_node_id].add(modality)
                entity_to_chunks[ent_node_id].add(chunk_id)

                ent_vdb_id = _md5(ent_node_id)
                new_entity_texts.append(entity_name)
                new_entity_ids.append((ent_vdb_id, ent_node_id, source, entity_name_zh))

    # Step 5: embed text VDBs (embedding API calls — outside lock)
    entities_added = _upsert_text_vdb(
        vdb_entity, new_entity_texts, new_entity_ids, embedder, key="entity_id",
        _lock=_lock,
    )
    hyperedges_added = _upsert_text_vdb(
        vdb_hyperedge, new_hyperedge_texts, new_hyperedge_ids, embedder, key="hyperedge_id",
        _lock=_lock,
    )

    # Step 6: asset nodes — under lock
    asset_added = 0
    with _ctx:
        if media_records:
            asset_added = _add_asset_nodes(
                media_records=media_records,
                chunks=chunks,
                graph=graph,
                entity_to_chunks=entity_to_chunks,
                source_name=source_name,
            )
            for r in media_records:
                file_modalities.add(_media_modality(r))

        # Step 7: cross-modal hyperedges
        cm_added = _add_cross_modal_hyperedges(
            graph=graph,
            entity_to_modalities=entity_to_modalities,
            source_name=source_name,
        )

    # Step 8: embed media descriptions into media VDB
    media_added = 0
    if media_records and vdb_media is not None and media_embedder is not None:
        media_added = _upsert_media_vdb(vdb_media, media_records, media_embedder, _lock=_lock)

    logger.info(
        "File %s: %d chunks, %d entities, %d hyperedges, %d cross-modal, "
        "%d assets, %d media",
        source_name, len(chunks), entities_added, hyperedges_added,
        cm_added, asset_added, media_added,
    )
    return {
        "chunks": len(chunks),
        "entities": entities_added,
        "hyperedges": hyperedges_added,
        "media_records": media_added,
        "cross_modal_edges": cm_added,
        "asset_nodes": asset_added,
        "modalities": file_modalities,
    }


# ---------------------------------------------------------------------------
# Asset / cross-modal helpers
# ---------------------------------------------------------------------------

def _media_modality(record: Dict) -> str:
    et = record.get("embed_type", "media")
    if et == "image":
        return "image"
    if et == "audio":
        return "audio"
    if et in ("video", "video_segment"):
        return "video"
    return "media"


def _add_asset_nodes(
    media_records: List[Dict],
    chunks: List[Dict],
    graph: GraphStorage,
    entity_to_chunks: Dict[str, Set[str]],
    source_name: str,
) -> int:
    """Create one asset node per media record and link it to entities sharing
    its chunk_id."""
    chunk_to_entities: Dict[str, List[str]] = defaultdict(list)
    for ent, cids in entity_to_chunks.items():
        for cid in cids:
            chunk_to_entities[cid].append(ent)

    added = 0
    for r in media_records:
        modality = _media_modality(r)
        path = (
            r.get("image_path")
            or r.get("audio_path")
            or r.get("clip_path")
            or r.get("source")
            or source_name
        )
        node_id = asset_node_id(modality, path)
        graph.upsert_node(node_id, {
            "role": "asset",
            "modality": modality,
            "path": path,
            "source": source_name,
            "sources": source_name,
            "modalities": modality,
            "description": (r.get("description") or r.get("transcript") or "")[:500],
        })
        added += 1

        # Link the asset to the entities that share its chunk
        cid = r.get("chunk_id", "")
        if cid and cid in chunk_to_entities:
            for ent in chunk_to_entities[cid]:
                graph.upsert_edge(node_id, ent, {"kind": "asset_mentions"})
                # also tag the entity with the asset's modality
                graph.upsert_node(ent, {"modalities": modality, "sources": source_name})

    return added


def _add_cross_modal_hyperedges(
    graph: GraphStorage,
    entity_to_modalities: Dict[str, Set[str]],
    source_name: str,
) -> int:
    """For every entity that appears in chunks of more than one modality
    *within this source*, create a cross-modal hyperedge linking it back to
    itself with the union of modalities recorded as an attribute."""
    added = 0
    for ent_id, mods in entity_to_modalities.items():
        if len(mods) < 2:
            continue
        mods_sorted = sorted(mods)
        xm_id = cross_modal_hyperedge_id(source_name, ent_id, mods_sorted)
        sentence = (
            f"Cross-modal coreference of {ent_id} across "
            f"{', '.join(mods_sorted)} within {source_name}"
        )
        graph.upsert_node(xm_id, {
            "role": "hyperedge",
            "kind": "cross_modal",
            "modality": "cross_modal",
            "modalities": GRAPH_FIELD_SEP.join(mods_sorted),
            "source": source_name,
            "sources": source_name,
            "sentence": sentence,
        })
        graph.upsert_edge(xm_id, ent_id, {"kind": "cross_modal_link"})
        added += 1
    return added


# ---------------------------------------------------------------------------
# Vector DB helpers
# ---------------------------------------------------------------------------

def _upsert_text_vdb(
    vdb: VectorStorage,
    texts: List[str],
    ids: List[Tuple],
    embedder,
    key: str,
    _lock: Optional[threading.Lock] = None,
) -> int:
    if not texts:
        return 0
    # Embedding call — outside lock (network I/O)
    vectors = embedder.embed(texts)
    batch_entries = []
    seen_ids: Set[str] = set()
    for vec, id_tuple, txt in zip(vectors, ids, texts):
        vid, nid, src = id_tuple[0], id_tuple[1], id_tuple[2]
        name_zh = id_tuple[3] if len(id_tuple) > 3 else ""
        if vid in seen_ids:
            continue
        seen_ids.add(vid)
        meta = {"content": txt, key: nid, "source": src}
        if name_zh and name_zh != txt:
            meta["name_zh"] = name_zh
        batch_entries.append({"id": vid, "vector": vec, "meta": meta})
    # VDB write — under lock
    if _lock:
        with _lock:
            vdb.upsert_batch(batch_entries)
    else:
        vdb.upsert_batch(batch_entries)
    return len(batch_entries)


def _upsert_media_vdb(
    vdb_media: VectorStorage,
    media_records: List[Dict],
    embedder,
    _lock: Optional[threading.Lock] = None,
) -> int:
    if not media_records:
        return 0

    texts = []
    for r in media_records:
        text = r.get("description") or r.get("transcript") or r.get("content") or ""
        texts.append(text.strip() or " ")

    try:
        vectors = embedder.embed(texts)
    except Exception as exc:
        logger.warning("Media embedding failed: %s", exc)
        return 0

    entries = []
    for r, vec, txt in zip(media_records, vectors, texts):
        record_id = r.get("image_id") or r.get("audio_id") or r.get("segment_id") or _md5(txt)
        embed_type = r.get("embed_type", "media")
        entries.append({
            "id": record_id,
            "vector": vec,
            "meta": {
                "content": txt,
                "source": r.get("source", ""),
                "chunk_id": r.get("chunk_id", ""),
                "embed_type": embed_type,
                "modality": _media_modality(r),
            },
        })

    if _lock:
        with _lock:
            vdb_media.upsert_batch(entries)
    else:
        vdb_media.upsert_batch(entries)
    return len(entries)
