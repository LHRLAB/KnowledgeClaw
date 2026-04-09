"""
Multimodal client for transcription and captioning via apiyi + Gemini.

Uses the same OpenAI-compatible apiyi proxy as the rest of KnowledgeClaw.
No local GPU server required.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_CLIENT: Optional["OmniClient"] = None


def get_omni_client() -> "OmniClient":
    global _CLIENT
    if _CLIENT is None:
        import config as cfg
        _CLIENT = OmniClient(
            api_key=cfg.OPENAI_API_KEY,
            base_url=cfg.OPENAI_BASE_URL,
            model=cfg.OMNI_MODEL,
        )
    return _CLIENT


def _to_data_url(file_path: str, mime_map: dict, default_mime: str) -> str:
    ext = os.path.splitext(file_path)[-1].lstrip(".").lower()
    mime = mime_map.get(ext, default_mime)
    with open(file_path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:{mime};base64,{data}"


_AUDIO_MIME = {
    "mp3": "audio/mpeg", "wav": "audio/wav", "flac": "audio/flac",
    "m4a": "audio/mp4", "ogg": "audio/ogg",
}
_IMAGE_MIME = {
    "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
    "bmp": "image/bmp", "webp": "image/webp",
}


class OmniClient:
    """
    Wrapper around apiyi-proxied Gemini for transcription and captioning.
    Sends audio/image as base64 data URLs via OpenAI-compatible chat API.
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def _chat(self, content: list, max_tokens: int = 1024) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def transcribe(self, audio_path: str) -> str:
        """Transcribe an audio file."""
        url = _to_data_url(audio_path, _AUDIO_MIME, "audio/wav")
        return self._chat([
            {"type": "input_audio", "input_audio": {"data": url.split(",", 1)[1], "format": "wav"}},
            {"type": "text", "text": "Please transcribe this audio accurately. Output only the transcript."},
        ], max_tokens=2048)

    def caption_image(self, image_path: str, context: str = "") -> str:
        """Describe an image."""
        prompt = "Describe this image in detail."
        if context:
            prompt = f"Context: '{context[:200]}'. Describe this image in detail."
        url = _to_data_url(image_path, _IMAGE_MIME, "image/jpeg")
        return self._chat([
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": prompt},
        ])

    def caption_audio_segment(self, audio_path: str, transcript: str,
                               start: float, end: float) -> str:
        """Describe the content and context of an audio segment."""
        prompt = (
            f"Audio segment [{start:.1f}s-{end:.1f}s]. "
            f"Transcript: '{transcript[:300]}'. "
            "Describe the content and context of this audio in detail."
        )
        # Fall back to text-only if audio upload not supported
        try:
            url = _to_data_url(audio_path, _AUDIO_MIME, "audio/wav")
            return self._chat([
                {"type": "input_audio", "input_audio": {"data": url.split(",", 1)[1], "format": "wav"}},
                {"type": "text", "text": prompt},
            ])
        except Exception:
            return self._chat([{"type": "text", "text": prompt}])

    def caption_video_frame(self, frame_path: str, transcript: str,
                             start: float, end: float) -> str:
        """Describe a video keyframe."""
        prompt = (
            f"Keyframe from video segment [{start:.1f}s-{end:.1f}s]. "
            f"Transcript: '{transcript[:300]}'. "
            "Describe what you see and relate it to the spoken content."
        )
        url = _to_data_url(frame_path, _IMAGE_MIME, "image/jpeg")
        return self._chat([
            {"type": "image_url", "image_url": {"url": url}},
            {"type": "text", "text": prompt},
        ])
