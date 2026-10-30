"""The model backend KnowledgeClaw calls: one chat model (text + raw media parts) and one embedding model.

Every component goes through a Backend, so the same code runs standalone (OpenAIBackend, any
OpenAI-compatible endpoint) or inside an evaluation harness that meters and pins the models (the harness
passes its own object with the same two methods).
"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Protocol


class Backend(Protocol):
    def chat(self, messages: list[dict], *, json_mode: bool = True, max_tokens: int = 1024,
             temperature: float = 0.0):
        """JSON mode returns the parsed object (or None); otherwise the text."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


def parse_json(text: str):
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except Exception:
        for a, b in (("{", "}"), ("[", "]")):
            i, j = text.find(a), text.rfind(b)
            if i != -1 and j > i:
                try:
                    return json.loads(text[i:j + 1])
                except Exception:
                    pass
    return None


def image_part(data: bytes, mime: str) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode()}"}}


def audio_part(data: bytes, fmt: str = "wav") -> dict:
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": fmt}}


def file_part(path: Path) -> dict:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if mime.startswith("image/"):
        return image_part(path.read_bytes(), mime)
    return audio_part(path.read_bytes(), path.suffix.lstrip("."))


class OpenAIBackend:
    """Any OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, base_url: str, model: str, embed_model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model, self.embed_model = model, embed_model

    def chat(self, messages, *, json_mode=True, max_tokens=1024, temperature=0.0):
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            r = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=max_tokens,
                                                    temperature=temperature, **kw)
            if not getattr(r, "choices", None):
                raise RuntimeError("empty response from model")
            text = r.choices[0].message.content or ""
        except Exception:
            return None
        return parse_json(text) if json_mode else text

    def embed(self, texts):
        out = []
        for s in range(0, len(texts), 64):
            r = self.client.embeddings.create(model=self.embed_model, input=texts[s:s + 64])
            out += [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
        return out
