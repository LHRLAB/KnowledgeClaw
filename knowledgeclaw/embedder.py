"""
OpenAI embeddings via apiyi proxy.
Batches are sent in parallel to reduce wall-clock time on large inputs.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from openai import OpenAI

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100   # texts per API call
_MAX_WORKERS = 4    # parallel embedding requests


class Embedder:
    def __init__(self, api_key: str, base_url: str, model: str, dim: int):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dim = dim

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        batch = [t if t.strip() else " " for t in batch]
        response = self._client.embeddings.create(model=self._model, input=batch)
        return [d.embedding for d in sorted(response.data, key=lambda d: d.index)]

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts.
        Splits into batches of _BATCH_SIZE and sends them in parallel.
        Returns vectors in the same order as inputs.
        """
        if not texts:
            return []

        batches = [texts[i : i + _BATCH_SIZE] for i in range(0, len(texts), _BATCH_SIZE)]
        batch_results: List[List[List[float]]] = [None] * len(batches)  # type: ignore
        workers = min(len(batches), _MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=workers) as exe:
            future_to_idx = {exe.submit(self._embed_batch, b): i for i, b in enumerate(batches)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                batch_results[idx] = future.result()

        return [vec for batch in batch_results for vec in batch]

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
