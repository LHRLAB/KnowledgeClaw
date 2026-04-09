"""
LLM-based entity / proposition extractor.
Extracts propositions with entities; each entity carries both an English name
and a Chinese translation (name_zh).  Batches are processed in parallel via
ThreadPoolExecutor to avoid serial latency on large documents.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)

# Max parallel LLM calls during extract_batch
_MAX_WORKERS = 8

EXTRACT_PROMPT = """Extract key entities and propositions from the text below.
For each sentence/proposition identify the named entities (people, organisations, concepts, methods, tools, places).
For each entity provide:
  - "name": canonical English name, UPPER-CASED. If the entity is originally in Chinese or another non-English language, provide the standard English translation or transliteration.
  - "name_zh": Chinese name. If the entity is originally in Chinese, keep the original. If originally in English, provide the standard Chinese translation. For proper nouns or abbreviations with no natural Chinese equivalent, repeat the English form.

Return JSON ONLY (no markdown, no explanation):
{{
  "propositions": [
    {{
      "sentence": "atomic fact or claim",
      "entities": [
        {{"name": "ENTITY_EN", "name_zh": "实体中文名"}}
      ]
    }}
  ]
}}
Text: {text}"""


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


class LLMExtractor:
    """Uses an LLM to extract propositions and entity mentions from text chunks."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def _extract_one(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract propositions from a single text chunk.

        Returns list of:
          {"sentence": str, "entities": [{"name": str, "name_zh": str}]}
        """
        prompt = EXTRACT_PROMPT.format(text=text)
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=1024,
            )
            raw = resp.choices[0].message.content or ""
            raw = _clean_json(raw)
            data = json.loads(raw)
            propositions = data.get("propositions", [])
            validated = []
            for p in propositions:
                if not isinstance(p, dict) or "sentence" not in p:
                    continue
                raw_entities = p.get("entities", [])
                entities: List[Dict[str, str]] = []
                for e in raw_entities:
                    if isinstance(e, dict):
                        name = str(e.get("name", "")).strip()
                        name_zh = str(e.get("name_zh", name)).strip()
                    else:
                        # Backward-compat: plain string entity
                        name = str(e).strip()
                        name_zh = name
                    if name:
                        entities.append({"name": name, "name_zh": name_zh})
                validated.append({
                    "sentence": str(p["sentence"]).strip(),
                    "entities": entities,
                })
            return validated
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error in extractor: %s", exc)
            return []
        except Exception as exc:
            logger.error("LLMExtractor error: %s", exc)
            return []

    def extract_batch(self, texts: List[str]) -> List[List[Dict[str, Any]]]:
        """
        Extract propositions from each text in parallel.

        Returns a list (same length as texts) of proposition lists.
        """
        if not texts:
            return []

        results: List[List[Dict[str, Any]]] = [[] for _ in texts]
        workers = min(len(texts), _MAX_WORKERS)

        with ThreadPoolExecutor(max_workers=workers) as exe:
            future_to_idx = {exe.submit(self._extract_one, t): i for i, t in enumerate(texts)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error("extract_batch worker %d failed: %s", idx, exc)

        return results
