"""The frozen model stack every run uses: one chat model and one encoder behind an OpenAI-compatible endpoint.

Calls are cached on disk (keyed by the full request), so a re-run or a re-score never pays twice and an
interrupted run resumes where it stopped.

    MODEL_API_KEY, MODEL_BASE_URL     endpoint (default: OpenRouter)
    EVAL_LLM, EVAL_EMBED              google/gemini-2.5-flash-lite, google/gemini-embedding-2-preview
    CACHE_DIR                         default results/.cache
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path

from knowledgeclaw.backend import parse_json

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get("MODEL_BASE_URL", "https://openrouter.ai/api/v1")
EVAL_LLM = os.environ.get("EVAL_LLM", "google/gemini-2.5-flash-lite")
EVAL_EMBED = os.environ.get("EVAL_EMBED", "google/gemini-embedding-2-preview")
CACHE = Path(os.environ.get("CACHE_DIR", ROOT / "results" / ".cache"))

_client = None
_lock = threading.Lock()


def client():
    global _client
    with _lock:
        if _client is None:
            from openai import OpenAI
            key = os.environ.get("MODEL_API_KEY") or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            _client = OpenAI(api_key=key, base_url=BASE_URL, max_retries=0, timeout=300.0)
    return _client


def _path(kind: str, key: str) -> Path:
    p = CACHE / kind / key[:2]
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{key}.json"


def _key(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _read(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write(p: Path, obj) -> None:
    tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    tmp.replace(p)


def _retry(fn, attempts: int = 6):
    last = None
    for a in range(attempts):
        try:
            return fn()
        except Exception as e:                      # rate limit / network: back off and retry
            last = e
            time.sleep(min(2 ** a, 30) + random.random())
    raise RuntimeError(f"model call failed after retries: {last!r}")


def chat_text(messages: list[dict], *, json_mode: bool = True, max_tokens: int = 1024,
              temperature: float = 0.0) -> str | None:
    """Raw completion text (cached); None if the endpoint keeps failing."""
    body = {"model": EVAL_LLM, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    cp = _path("chat", _key(body))
    hit = _read(cp)
    if hit is not None:
        return hit["text"]
    extra = {"provider": {"allow_fallbacks": True, "require_parameters": True}} if "openrouter" in BASE_URL else None
    try:
        def _call():
            resp = client().chat.completions.create(**body, extra_body=extra)
            if not getattr(resp, "choices", None):
                raise RuntimeError("empty response from model")
            return resp
        r = _retry(_call)
        text = r.choices[0].message.content or ""
    except Exception as e:
        print(f"[llm] chat failed: {e!r}", flush=True)
        return None
    _write(cp, {"text": text})
    return text


def chat(messages: list[dict], *, json_mode: bool = True, max_tokens: int = 1024, temperature: float = 0.0):
    text = chat_text(messages, json_mode=json_mode, max_tokens=max_tokens, temperature=temperature)
    if text is None:
        return None
    return parse_json(text) if json_mode else text


def embed(texts: list[str], batch: int = 64) -> list[list[float]]:
    """Embeddings, cached per text."""
    texts = [t if t and t.strip() else "(empty)" for t in texts]
    out: list = [None] * len(texts)
    todo = []
    for i, t in enumerate(texts):
        hit = _read(_path("embed", _key([EVAL_EMBED, t])))
        if hit is not None:
            out[i] = hit["v"]
        else:
            todo.append(i)
    for s in range(0, len(todo), batch):
        idx = todo[s:s + batch]
        r = _retry(lambda: client().embeddings.create(model=EVAL_EMBED, input=[texts[i] for i in idx]))
        for i, d in zip(idx, sorted(r.data, key=lambda d: d.index)):
            out[i] = d.embedding
            _write(_path("embed", _key([EVAL_EMBED, texts[i]])), {"v": d.embedding})
    return out


class Backend:
    """The knowledgeclaw.Backend the evaluation hands to the method."""

    def chat(self, messages, *, json_mode=True, max_tokens=1024, temperature=0.0):
        return chat(messages, json_mode=json_mode, max_tokens=max_tokens, temperature=temperature)

    def embed(self, texts):
        return embed(texts)
