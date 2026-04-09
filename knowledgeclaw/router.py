"""
Query intent router (C3) for KnowledgeClaw.

Routes a user query to one of three retrieval modes:

  * "subgraph"  → fine-grained subgraph retrieval (precise factual lookups)
  * "community" → community-level retrieval (open-ended thematic synthesis)
  * "hybrid"    → both tiers concatenated (the safest default for ambiguous
                  queries; the QA agent dedupes overlapping evidence)

The router uses a fast heuristic by default and consults the LLM only when
the heuristic is not confident — this keeps latency low for the common case.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Heuristic vocabulary for the two extremes.
_SYNTHESIS_KEYWORDS = (
    "summarize", "summarise", "summary", "overview", "across",
    "overall", "themes", "theme", "general", "in general",
    "what do i know about", "what does my kb know about",
    "main topics", "broad picture", "synthesis", "synthesize",
    "synthesise", "trends", "patterns", "compare all", "all sources",
    "总结", "综述", "概览", "整体", "主题", "梳理",
)

_FACTUAL_KEYWORDS = (
    "when did", "who said", "where is", "what is the value", "how much",
    "what time", "which page", "exact", "specifically", "the precise",
    "在哪", "几点", "多少", "哪一页", "原文",
)

# Question-mark follow-up cue: very short questions tend to be factual.
_SHORT_QUERY_TOKEN_THRESHOLD = 6


class IntentRouter:
    """Decide which retrieval tier(s) to use for a given query."""

    LLM_PROMPT = """Classify the following user query into ONE label that describes the BEST retrieval strategy over a personal knowledge hypergraph.

Labels:
  - subgraph  : precise factual lookup, single fact, definition, who/when/where/which
  - community : open-ended synthesis, summarisation, "what do I know about X", thematic overviews
  - hybrid    : ambiguous or multi-hop questions that need both precise facts and broader context

Reply with EXACTLY one of: subgraph, community, hybrid

Query: {query}
Label:"""

    def __init__(self, llm_client=None, model: Optional[str] = None, use_llm: bool = True):
        self._client = llm_client
        self._model = model
        self._use_llm = use_llm and llm_client is not None and model is not None

    # ---------------------------------------------------------------- public

    def route(self, query: str) -> str:
        """Return one of {'subgraph', 'community', 'hybrid'}."""
        q = (query or "").strip().lower()
        if not q:
            return "subgraph"

        # 1. Heuristic — fast & deterministic
        heuristic = self._heuristic(q)
        if heuristic != "hybrid":
            return heuristic

        # 2. LLM fallback — only when heuristic is unsure
        if self._use_llm:
            llm_label = self._llm_route(query)
            if llm_label in ("subgraph", "community", "hybrid"):
                return llm_label
        return "hybrid"

    # ---------------------------------------------------------------- helpers

    def _heuristic(self, q: str) -> str:
        if any(kw in q for kw in _SYNTHESIS_KEYWORDS):
            return "community"
        if any(kw in q for kw in _FACTUAL_KEYWORDS):
            return "subgraph"
        if q.endswith("?") and len(re.findall(r"\w+", q)) <= _SHORT_QUERY_TOKEN_THRESHOLD:
            return "subgraph"
        return "hybrid"

    def _llm_route(self, query: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": self.LLM_PROMPT.format(query=query)}],
                temperature=0,
                max_tokens=8,
            )
            text = (resp.choices[0].message.content or "").strip().lower()
            for label in ("subgraph", "community", "hybrid"):
                if label in text:
                    return label
        except Exception as exc:
            logger.warning("Intent router LLM call failed: %s", exc)
        return "hybrid"
