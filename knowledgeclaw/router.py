"""C3, the intent router: rho(q) in {subgraph, community, hybrid}.

Anchored queries (the question names a file or an entity present in K) always go to the subgraph tier.
Otherwise the router is heuristic first -- synthesis cues select the community tier, lookup cues or a short
question the subgraph tier -- and only when neither fires does one short model call classify the query;
a failed call falls back to hybrid.
"""
from __future__ import annotations

import re

SYNTHESIS = ("summarize", "summarise", "summary", "overview", "across", "overall", "themes", "theme",
             "in general", "what do i know about", "main topics", "big picture", "synthesis", "synthesize",
             "synthesise", "trends", "patterns", "compare all", "all sources")
LOOKUP = ("when did", "who ", "where is", "what is the", "how much", "how many", "what time", "which page",
          "exact", "date of", "title of", "in which")
SHORT = 6

PROMPT = """Classify the query into ONE retrieval strategy over a personal knowledge hypergraph.
subgraph: a precise fact, definition, who/when/where/which. community: open-ended synthesis or a thematic \
overview. hybrid: ambiguous, or needs both.
Reply with exactly one word: subgraph, community or hybrid.
Query: {q}"""


def route(q: str, anchored: bool, llm=None) -> str:
    if anchored:
        return "subgraph"
    s = q.strip().lower()
    if any(k in s for k in SYNTHESIS):
        return "community"
    if any(k in s for k in LOOKUP) or (s.endswith("?") and len(re.findall(r"\w+", s)) <= SHORT):
        return "subgraph"
    if llm is not None:
        r = llm.chat([{"role": "user", "content": PROMPT.format(q=q)}], json_mode=False, max_tokens=5) or ""
        for lab in ("subgraph", "community", "hybrid"):
            if lab in r.lower():
                return lab
    return "hybrid"
