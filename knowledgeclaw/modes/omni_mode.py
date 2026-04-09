"""
Video mode processor (adapted from OmniGraph-Agent).

Input:  Video file (mp4, avi, mov, mkv, …).
Output: text chunks (from Qwen2.5-Omni-3B transcript) + video-segment records.

Pipeline:
  1. Fixed-duration segmentation via ffprobe/ffmpeg
  2. Per-segment: extract video clip via ffmpeg → expr/ folder  [parallel]
  3. Per-segment: OmniClient transcribes audio from clip        [parallel]
  4. Per-segment: OmniClient captions segment                   [parallel]
  5. Segment records carry clip_path for native video embedding by e5-omni-3B
"""
from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from knowledgeclaw.chunker import chunk_text

logger = logging.getLogger(__name__)

DEFAULT_SEGMENT_DURATION = 30.0


def _mdhash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class OmniModeProcessor:

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
        Returns (text_chunks, segment_records).

        text_chunks: KnowledgeClaw chunk format
            {"content", "source", "source_path", "type", "chunk_index", "tokens"}
        segment_records: media records for e5-omni-3B embedding
            {"segment_id", "clip_path", "start_time", "end_time",
             "transcript", "description", "source", "chunk_id", "embed_type"}
        """
        source = os.path.basename(input_path)
        os.makedirs(self.expr_dir, exist_ok=True)

        duration = self._get_duration(input_path)
        if duration is None:
            logger.error("[Omni mode] Cannot determine duration of %s", source)
            return [], []

        logger.info("[Omni mode] %s: %.1fs, segmenting ...", source, duration)
        segs = self._build_segments(duration)
        n = len(segs)

        # Step 1: extract all clips in parallel
        clip_paths: List[Optional[str]] = [None] * n

        def _extract(i):
            s, e = segs[i]
            return i, self._extract_clip(input_path, source, s, e)

        with ThreadPoolExecutor(max_workers=self.seg_workers) as ex:
            for i, path in ex.map(_extract, range(n)):
                clip_paths[i] = path

        # Step 2: transcribe all clips in parallel
        transcripts: List[str] = [""] * n

        def _transcribe(i):
            cp = clip_paths[i]
            if cp is None:
                return i, ""
            s, e = segs[i]
            try:
                return i, self._transcribe_segment(cp, source, s, e)
            except Exception as exc:
                logger.warning("[Omni mode] Transcription failed [%.1fs-%.1fs]: %s", s, e, exc)
                return i, ""

        with ThreadPoolExecutor(max_workers=self.seg_workers) as ex:
            for i, tr in ex.map(_transcribe, range(n)):
                transcripts[i] = tr

        # Step 3: caption all clips in parallel
        descriptions: List[str] = [f"[{s:.1f}s-{e:.1f}s]" for s, e in segs]

        def _caption(i):
            cp = clip_paths[i]
            if cp is None:
                return i, descriptions[i]
            s, e = segs[i]
            try:
                return i, self._caption_segment(cp, transcripts[i], s, e)
            except Exception as exc:
                logger.warning("[Omni mode] Caption failed [%.1fs-%.1fs]: %s", s, e, exc)
                return i, f"[{s:.1f}s-{e:.1f}s] {transcripts[i][:200]}"

        with ThreadPoolExecutor(max_workers=self.seg_workers) as ex:
            for i, desc in ex.map(_caption, range(n)):
                descriptions[i] = desc

        # Step 4: build chunks + segment records
        chunks: List[Dict] = []
        segment_records: List[Dict] = []
        chunk_counter = [0]

        for i, (start, end) in enumerate(segs):
            clip_path = clip_paths[i]
            if clip_path is None:
                continue
            transcript = transcripts[i]
            description = descriptions[i]

            seg_chunks = []
            if transcript.strip():
                for ct in chunk_text(transcript, max_tokens=self.chunk_size, overlap=self.overlap_size):
                    seg_chunks.append({
                        "content": ct["content"],
                        "source": source,
                        "source_path": str(Path(input_path).resolve()),
                        "type": "video",
                        "chunk_index": chunk_counter[0],
                        "tokens": ct["tokens"],
                        "start_time": start,
                        "end_time": end,
                    })
                    chunk_counter[0] += 1
            chunks.extend(seg_chunks)

            seg_id = _mdhash(f"{source}_{start:.1f}_{end:.1f}")
            last_chunk_id = ""
            if seg_chunks:
                last_chunk_id = _mdhash(seg_chunks[-1]["content"] + " | Source: " + source)

            segment_records.append({
                "segment_id": seg_id,
                "clip_path": clip_path,
                "start_time": start,
                "end_time": end,
                "transcript": transcript,
                "description": description,
                "source": source,
                "chunk_id": last_chunk_id,
                "embed_type": "video_segment",
            })

        logger.info(
            "[Omni mode] %s: %d text chunks, %d video segments",
            source, len(chunks), len(segment_records),
        )
        return chunks, segment_records

    # ── Helpers ─────────────────────────────────────────────────────────────── #

    def _get_duration(self, video_path: str) -> Optional[float]:
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip())
        except Exception as e:
            logger.warning("Cannot get video duration via ffprobe: %s", e)
            return None

    def _build_segments(self, duration: float) -> List[Tuple[float, float]]:
        segs = []
        start = 0.0
        step = max(self.segment_duration - self.segment_overlap, 1.0)
        while start < duration:
            end = min(start + self.segment_duration, duration)
            segs.append((start, end))
            start += step
        return segs

    def _extract_clip(self, video_path: str, source: str,
                      start: float, end: float) -> Optional[str]:
        stem = os.path.splitext(source)[0]
        clip_name = f"{stem}_{start:.1f}s_{end:.1f}s.mp4"
        clip_path = os.path.join(self.expr_dir, clip_name)
        if os.path.exists(clip_path):
            return clip_path
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-t", str(end - start),
                    "-i", video_path,
                    "-c:v", "libx264", "-c:a", "aac",
                    "-preset", "ultrafast",
                    clip_path,
                ],
                capture_output=True, check=True,
            )
            return clip_path
        except Exception as e:
            logger.warning("ffmpeg clip extract failed [%.1fs-%.1fs]: %s", start, end, e)
            return None

    def _extract_audio_from_clip(self, clip_path: str) -> Optional[str]:
        try:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.close()
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", clip_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "16000",
                    tmp.name,
                ],
                capture_output=True, check=True,
            )
            return tmp.name
        except Exception as e:
            logger.warning("ffmpeg audio extract failed: %s", e)
            return None

    def _transcribe_segment(self, clip_path: str, source: str,
                             start: float, end: float) -> str:
        audio_file = self._extract_audio_from_clip(clip_path)
        if audio_file is None:
            return ""
        try:
            from knowledgeclaw.serving.omni_client import get_omni_client
            return get_omni_client().transcribe(audio_file)
        except Exception as e:
            logger.warning("Transcription failed for %s [%.1fs-%.1fs]: %s", source, start, end, e)
            return ""
        finally:
            if audio_file and os.path.exists(audio_file):
                os.unlink(audio_file)

    def _caption_segment(self, clip_path: str, transcript: str,
                          start: float, end: float) -> str:
        audio_file = self._extract_audio_from_clip(clip_path)
        if audio_file:
            try:
                from knowledgeclaw.serving.omni_client import get_omni_client
                return get_omni_client().caption_audio_segment(audio_file, transcript, start, end)
            except Exception as e:
                logger.warning("Video caption failed: %s, using transcript", e)
            finally:
                if os.path.exists(audio_file):
                    os.unlink(audio_file)
        return f"[{start:.1f}s-{end:.1f}s] {transcript[:200]}"
