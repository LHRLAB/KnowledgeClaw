"""
Audio mode processor (adapted from OmniGraph-Agent).

Input:  Audio file (mp3, wav, flac, etc.)
Output: text chunks (from Qwen2.5-Omni-3B transcript) + audio segment records.

Pipeline:
  1. Fixed-duration segmentation via soundfile
  2. Per-segment: save audio slice to expr/ folder           [parallel]
  3. Per-segment: OmniClient transcribes → text chunks       [parallel]
  4. Per-segment: OmniClient captions → description          [parallel]
  5. Audio records carry segment path for e5-omni-3B embedding
"""
from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

from knowledgeclaw.chunker import chunk_text

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_DURATION = 30.0


def _mdhash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _get_audio_duration(audio_path: str) -> float:
    import soundfile as sf
    info = sf.info(audio_path)
    return info.duration


def _extract_segment(audio_path: str, start: float, end: float, out_path: str) -> str:
    import soundfile as sf
    data, sr = sf.read(audio_path, always_2d=False)
    s = int(start * sr)
    e = int(min(end * sr, len(data)))
    segment = data[s:e]
    sf.write(out_path, segment, sr)
    return out_path


class AudioModeProcessor:

    def __init__(
        self,
        chunk_size: int = 512,
        overlap_size: int = 64,
        segment_duration: float = DEFAULT_SEGMENT_DURATION,
        segment_overlap: float = 0.0,
        seg_workers: int = 4,
        expr_dir: str = "./expr",
        **kwargs,
    ):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self.segment_duration = segment_duration
        self.segment_overlap = segment_overlap
        self.seg_workers = seg_workers
        self.expr_dir = expr_dir

    def process(self, input_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Returns (text_chunks, audio_records).

        text_chunks: KnowledgeClaw chunk format
        audio_records: media records for e5-omni-3B embedding
            {"audio_id", "audio_path", "transcript", "description",
             "source", "chunk_id", "start_time", "end_time", "embed_type"}
        """
        source = os.path.basename(input_path)
        os.makedirs(self.expr_dir, exist_ok=True)

        logger.info("[Audio mode] Processing %s ...", source)

        try:
            duration = _get_audio_duration(input_path)
        except Exception as e:
            logger.error("[Audio mode] Cannot read %s: %s", source, e)
            return [], []

        stem = os.path.splitext(source)[0]
        segs = self._build_segments(duration)
        n = len(segs)

        seg_paths: List[str] = []
        for seg_start, seg_end in segs:
            seg_name = f"{stem}_{seg_start:.1f}s_{seg_end:.1f}s.wav"
            seg_paths.append(os.path.join(self.expr_dir, seg_name))

        # Step 1: extract all segments in parallel
        def _extract_one(i):
            seg_start, seg_end = segs[i]
            seg_path = seg_paths[i]
            if not os.path.exists(seg_path):
                try:
                    _extract_segment(input_path, seg_start, seg_end, seg_path)
                except Exception as exc:
                    logger.warning(
                        "[Audio mode] Extract failed [%.1fs-%.1fs]: %s", seg_start, seg_end, exc
                    )
                    return i, False
            return i, True

        extract_ok = [True] * n
        with ThreadPoolExecutor(max_workers=self.seg_workers) as ex:
            for i, ok in ex.map(_extract_one, range(n)):
                extract_ok[i] = ok

        # Step 2: transcribe all in parallel
        transcripts: List[str] = [""] * n

        def _transcribe_one(i):
            if not extract_ok[i]:
                return i, ""
            seg_start, seg_end = segs[i]
            try:
                return i, self._transcribe(seg_paths[i], source, seg_start, seg_end)
            except Exception as exc:
                logger.warning(
                    "[Audio mode] Transcription failed [%.1fs-%.1fs]: %s", seg_start, seg_end, exc
                )
                return i, ""

        with ThreadPoolExecutor(max_workers=self.seg_workers) as ex:
            for i, tr in ex.map(_transcribe_one, range(n)):
                transcripts[i] = tr

        # Step 3: caption all in parallel
        descriptions: List[str] = [f"[{s:.1f}s-{e:.1f}s]" for s, e in segs]

        def _caption_one(i):
            if not extract_ok[i]:
                return i, descriptions[i]
            seg_start, seg_end = segs[i]
            try:
                return i, self._caption(seg_paths[i], transcripts[i], seg_start, seg_end)
            except Exception as exc:
                logger.warning(
                    "[Audio mode] Caption failed [%.1fs-%.1fs]: %s", seg_start, seg_end, exc
                )
                return i, f"[{seg_start:.1f}s-{seg_end:.1f}s] {transcripts[i][:200]}"

        with ThreadPoolExecutor(max_workers=self.seg_workers) as ex:
            for i, desc in ex.map(_caption_one, range(n)):
                descriptions[i] = desc

        # Step 4: build chunks + audio records
        chunks: List[Dict] = []
        audio_records: List[Dict] = []
        chunk_counter = [0]

        for i, (seg_start, seg_end) in enumerate(segs):
            if not extract_ok[i]:
                continue
            transcript = transcripts[i]
            description = descriptions[i]
            seg_path = seg_paths[i]

            seg_chunks = []
            if transcript.strip():
                for ct in chunk_text(transcript, max_tokens=self.chunk_size, overlap=self.overlap_size):
                    seg_chunks.append({
                        "content": ct["content"],
                        "source": source,
                        "source_path": str(Path(input_path).resolve()),
                        "type": "audio",
                        "chunk_index": chunk_counter[0],
                        "tokens": ct["tokens"],
                        "start_time": seg_start,
                        "end_time": seg_end,
                    })
                    chunk_counter[0] += 1
            chunks.extend(seg_chunks)

            audio_id = _mdhash(f"{source}_{seg_start:.1f}_{seg_end:.1f}")
            last_chunk_id = ""
            if seg_chunks:
                last_chunk_id = _mdhash(seg_chunks[-1]["content"] + " | Source: " + source)

            audio_records.append({
                "audio_id": audio_id,
                "audio_path": seg_path,
                "transcript": transcript,
                "description": description,
                "source": source,
                "chunk_id": last_chunk_id,
                "start_time": seg_start,
                "end_time": seg_end,
                "embed_type": "audio",
            })

        logger.info(
            "[Audio mode] %s: %d text chunks, %d audio segments",
            source, len(chunks), len(audio_records),
        )
        return chunks, audio_records

    def _build_segments(self, duration: float) -> List[Tuple[float, float]]:
        segs = []
        start = 0.0
        step = max(self.segment_duration - self.segment_overlap, 1.0)
        while start < duration:
            end = min(start + self.segment_duration, duration)
            segs.append((start, end))
            start += step
        return segs

    def _transcribe(self, seg_path: str, source: str, start: float, end: float) -> str:
        try:
            from knowledgeclaw.serving.omni_client import get_omni_client
            return get_omni_client().transcribe(seg_path)
        except Exception as e:
            logger.warning("Transcription failed for %s [%.1fs-%.1fs]: %s", source, start, end, e)
            return ""

    def _caption(self, seg_path: str, transcript: str, start: float, end: float) -> str:
        try:
            from knowledgeclaw.serving.omni_client import get_omni_client
            return get_omni_client().caption_audio_segment(seg_path, transcript, start, end)
        except Exception as e:
            logger.warning("Audio caption failed: %s, using transcript", e)
            return f"[{start:.1f}s-{end:.1f}s] {transcript[:200]}"
