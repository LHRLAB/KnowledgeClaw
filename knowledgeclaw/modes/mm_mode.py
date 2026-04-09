"""
PDF multimodal mode processor (adapted from OmniGraph-Agent).

Input:  PDF file.
Output: text chunks (one per page) + page image records.

Each PDF page is treated as one unit:
  - Page text → one chunk (feeds entity/hyperedge pipeline)
  - Page rendered as PNG image → one image record (embedded by e5-omni-3B)

This gives a 1:1 mapping between text chunks and page images.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def _mdhash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class MMModeProcessor:
    """
    Convert PDF pages into (text_chunk, page_image) pairs.
    Each page = one chunk + one image record.
    Page text serves as the caption/description for the page image.
    """

    def __init__(
        self,
        dpi: int = 200,
        expr_dir: str = "./expr",
        **kwargs,
    ):
        self.dpi = dpi
        self.expr_dir = expr_dir

    def process(self, input_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Returns (text_chunks, image_records).

        text_chunks: KnowledgeClaw chunk format, one per page
        image_records: media records for e5-omni-3B embedding
            {"image_id", "image_path", "context_text", "description",
             "source", "page", "chunk_id", "embed_type"}
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF is required: pip install pymupdf")

        source = os.path.basename(input_path)
        os.makedirs(self.expr_dir, exist_ok=True)

        doc = fitz.open(input_path)
        chunks: List[Dict] = []
        image_records: List[Dict] = []

        for page_num, page in enumerate(doc, start=1):
            # Extract page text
            page_text = page.get_text("text").strip()
            if not page_text:
                page_text = f"[Page {page_num} of {source} - no extractable text]"

            cid = _mdhash(
                page_text + " | Source: " + source + f" | Page: {page_num}"
            )
            chunks.append({
                "content": page_text,
                "source": source,
                "source_path": str(Path(input_path).resolve()),
                "type": "pdf",
                "chunk_index": page_num - 1,
                "tokens": len(page_text.split()),  # approximate
                "page": page_num,
                "chunk_id": cid,
            })

            # Render page as PNG image
            img_filename = f"{source}_p{page_num}.png"
            img_path = os.path.join(self.expr_dir, img_filename)

            zoom = self.dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            pix.save(img_path)

            image_id = _mdhash(img_path + " | Source: " + source)
            image_records.append({
                "image_id": image_id,
                "image_path": img_path,
                "context_text": page_text[:500],
                "description": page_text[:300],
                "source": source,
                "page": page_num,
                "chunk_id": cid,
                "embed_type": "image",
            })

        doc.close()
        logger.info(
            "[MM mode] %s: %d pages, %d page images",
            source, len(chunks), len(image_records),
        )
        return chunks, image_records
