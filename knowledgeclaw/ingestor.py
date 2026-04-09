"""
File ingestor: reads various file types and returns text chunks + optional media records.

Returns a tuple (chunks, media_records):
  - chunks: list of text chunk dicts for entity/hyperedge pipeline
  - media_records: list of media records for e5-omni-3B embedding (images/audio/video)
    empty list when omni server is not needed or unavailable
"""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

from knowledgeclaw.chunker import chunk_text

logger = logging.getLogger(__name__)

# File extension sets
_TEXT_EXTS = {".txt", ".md", ".rst", ".markdown", ".text"}
_CODE_EXTS = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp",
              ".cs", ".go", ".rb", ".php", ".swift", ".kt", ".rs", ".sh",
              ".bash", ".yaml", ".yml", ".toml", ".ini", ".cfg"}
_PDF_EXTS = {".pdf"}
_JSON_EXTS = {".json"}
_CSV_EXTS = {".csv"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
_AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv", ".flv"}

SUPPORTED_EXTENSIONS = (
    _TEXT_EXTS | _CODE_EXTS | _PDF_EXTS | _JSON_EXTS | _CSV_EXTS
    | _IMAGE_EXTS | _AUDIO_EXTS | _VIDEO_EXTS
)


def is_supported(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def ingest_file(
    file_path: str,
    max_tokens: int = 512,
    overlap: int = 64,
    expr_dir: str = "./expr",
) -> Tuple[List[Dict], List[Dict]]:
    """
    Read a file and return (chunks, media_records).

    chunks: [{"content", "source", "source_path", "type", "chunk_index", "tokens"}]
    media_records: [{"embed_type": "image"|"audio"|"video_segment", ...}]
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    source_name = path.name

    # ── Video ──────────────────────────────────────────────────────────────── #
    if ext in _VIDEO_EXTS:
        return _ingest_video(path, max_tokens, overlap, expr_dir)

    # ── Audio ──────────────────────────────────────────────────────────────── #
    if ext in _AUDIO_EXTS:
        return _ingest_audio(path, max_tokens, overlap, expr_dir)

    # ── PDF (multimodal: text + page images) ──────────────────────────────── #
    if ext in _PDF_EXTS:
        return _ingest_pdf(path, expr_dir)

    # ── Image ─────────────────────────────────────────────────────────────── #
    if ext in _IMAGE_EXTS:
        return _ingest_image(path, expr_dir)

    # ── Text types → text-only pipeline ───────────────────────────────────── #
    raw_text = ""
    file_type = "unknown"

    if ext in _TEXT_EXTS:
        raw_text = _read_text(path)
        file_type = "text"
    elif ext in _CODE_EXTS:
        raw_text = _read_code(path)
        file_type = "code"
    elif ext in _JSON_EXTS:
        raw_text = _read_json(path)
        file_type = "json"
    elif ext in _CSV_EXTS:
        raw_text = _read_csv(path)
        file_type = "csv"
    else:
        try:
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            file_type = "text"
        except Exception as exc:
            logger.warning("Cannot read %s: %s", file_path, exc)
            return [], []

    if not raw_text.strip():
        logger.info("Empty content from %s", file_path)
        return [], []

    base_chunks = chunk_text(raw_text, max_tokens=max_tokens, overlap=overlap)
    chunks = []
    for c in base_chunks:
        chunks.append({
            "content": c["content"],
            "source": source_name,
            "source_path": str(path.resolve()),
            "type": file_type,
            "chunk_index": c["chunk_index"],
            "tokens": c["tokens"],
        })
    return chunks, []


# ---------------------------------------------------------------------------
# Multimodal ingestors
# ---------------------------------------------------------------------------

def _ingest_video(path: Path, max_tokens: int, overlap: int,
                  expr_dir: str) -> Tuple[List[Dict], List[Dict]]:
    try:
        from knowledgeclaw.modes.omni_mode import OmniModeProcessor
        processor = OmniModeProcessor(
            chunk_size=max_tokens, overlap_size=overlap, expr_dir=expr_dir
        )
        return processor.process(str(path))
    except Exception as exc:
        logger.error("Video ingest failed for %s: %s", path.name, exc)
        return [], []


def _ingest_audio(path: Path, max_tokens: int, overlap: int,
                  expr_dir: str) -> Tuple[List[Dict], List[Dict]]:
    try:
        from knowledgeclaw.modes.audio_mode import AudioModeProcessor
        processor = AudioModeProcessor(
            chunk_size=max_tokens, overlap_size=overlap, expr_dir=expr_dir
        )
        return processor.process(str(path))
    except Exception as exc:
        logger.error("Audio ingest failed for %s: %s", path.name, exc)
        return [], []


def _ingest_pdf(path: Path, expr_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Use MMModeProcessor for PDFs: per-page text chunks + page images."""
    try:
        from knowledgeclaw.modes.mm_mode import MMModeProcessor
        processor = MMModeProcessor(expr_dir=expr_dir)
        return processor.process(str(path))
    except Exception as exc:
        logger.error("PDF ingest (MM mode) failed for %s: %s — falling back to text-only", path.name, exc)
        # Fallback: plain text extraction
        raw_text = _read_pdf_text(path)
        if not raw_text.strip():
            return [], []
        base_chunks = chunk_text(raw_text, max_tokens=512, overlap=64)
        chunks = [
            {
                "content": c["content"],
                "source": path.name,
                "source_path": str(path.resolve()),
                "type": "pdf",
                "chunk_index": c["chunk_index"],
                "tokens": c["tokens"],
            }
            for c in base_chunks
        ]
        return chunks, []


def _ingest_image(path: Path, expr_dir: str) -> Tuple[List[Dict], List[Dict]]:
    """Caption image with OmniClient + create image record for e5-omni-3B."""
    import hashlib

    source = path.name
    caption = ""
    try:
        from knowledgeclaw.serving.omni_client import get_omni_client
        caption = get_omni_client().caption_image(str(path))
    except Exception as exc:
        logger.warning("Image caption failed for %s: %s", source, exc)
        caption = f"[Image file: {source}]"

    chunks = []
    if caption.strip():
        base_chunks = chunk_text(caption, max_tokens=512, overlap=64)
        for c in base_chunks:
            chunks.append({
                "content": c["content"],
                "source": source,
                "source_path": str(path.resolve()),
                "type": "image",
                "chunk_index": c["chunk_index"],
                "tokens": c["tokens"],
            })

    image_id = hashlib.md5((str(path.resolve()) + " | Source: " + source).encode()).hexdigest()
    image_records = [{
        "image_id": image_id,
        "image_path": str(path.resolve()),
        "context_text": caption[:500],
        "description": caption[:300],
        "source": source,
        "chunk_id": "",
        "embed_type": "image",
    }]
    return chunks, image_records


# ---------------------------------------------------------------------------
# Text-only readers
# ---------------------------------------------------------------------------

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        logger.warning("Error reading text file %s: %s", path, exc)
        return ""


def _read_code(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
        lang = path.suffix.lstrip(".")
        return f"[Source code – language: {lang}, file: {path.name}]\n\n{content}"
    except Exception as exc:
        logger.warning("Error reading code file %s: %s", path, exc)
        return ""


def _read_pdf_text(path: Path) -> str:
    try:
        import fitz
        doc = fitz.open(str(path))
        pages = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages.append(f"[Page {page_num + 1}]\n{text}")
        doc.close()
        return "\n\n".join(pages)
    except Exception as exc:
        logger.warning("Error reading PDF %s: %s", path, exc)
        return ""


def _read_json(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return f"[JSON file: {path.name}]\n\n{json.dumps(data, ensure_ascii=False, indent=2)}"
    except Exception as exc:
        logger.warning("Error reading JSON %s: %s", path, exc)
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""


def _read_csv(path: Path) -> str:
    try:
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return ""

        header = rows[0]
        data_rows = rows[1:]
        chunk_size = 50
        parts = []
        header_md = "| " + " | ".join(header) + " |"
        sep_md = "| " + " | ".join(["---"] * len(header)) + " |"

        for i in range(0, max(1, len(data_rows)), chunk_size):
            batch = data_rows[i: i + chunk_size]
            lines = [header_md, sep_md]
            for row in batch:
                padded = list(row) + [""] * max(0, len(header) - len(row))
                lines.append(
                    "| " + " | ".join(str(c).replace("|", "\\|") for c in padded[: len(header)]) + " |"
                )
            parts.append("\n".join(lines))

        return f"[CSV file: {path.name}]\n\n" + "\n\n".join(parts)
    except Exception as exc:
        logger.warning("Error reading CSV %s: %s", path, exc)
        return ""
