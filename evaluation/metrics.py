"""Evaluation metrics for the KnowledgeClaw benchmark suite.

The two metrics used by PAPER.md Table 1:

* **Token-level F1** (SQuAD-style) — for QA splits.
* **Consistency Score** (0–100) — for the dynamic update split.

Both functions are pure and stateless so they can be reused by every
baseline as well as KnowledgeClaw itself.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List


# ---------------------------------------------------------------------------
# Token-level F1
# ---------------------------------------------------------------------------

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _normalise(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = _ARTICLES.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenise(text: str) -> List[str]:
    return _normalise(text).split()


def f1_score(pred: str, gold: str) -> float:
    """SQuAD-style token-level F1 between *pred* and *gold*."""
    pred_tokens = _tokenise(pred)
    gold_tokens = _tokenise(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_average(pairs: Iterable[tuple]) -> float:
    """Average F1 over an iterable of (pred, gold) pairs."""
    pairs = list(pairs)
    if not pairs:
        return 0.0
    total = sum(f1_score(p, g) for p, g in pairs)
    return total / len(pairs)


# ---------------------------------------------------------------------------
# Consistency score
# ---------------------------------------------------------------------------

def consistency_score(
    expected_state: dict,
    actual_state: dict,
) -> float:
    """Compare expected vs actual KB state after a sequence of update events.

    The expected state is a dict like::

        {
            "alive_sources":  set[str],   # files that should still be present
            "removed_sources": set[str],  # files that should be gone
            "alive_entities":  set[str],  # entities that must remain
        }

    Returns a 0-100 score that combines:
      - retention   (alive_* recall)
      - cleanliness (removed_* precision: nothing forbidden survived)
    """
    actual_sources = set(actual_state.get("alive_sources", []))
    actual_entities = set(actual_state.get("alive_entities", []))

    alive = set(expected_state.get("alive_sources", []))
    removed = set(expected_state.get("removed_sources", []))
    alive_ents = set(expected_state.get("alive_entities", []))

    retention = len(alive & actual_sources) / max(1, len(alive))
    forbidden_alive = len(removed & actual_sources)
    cleanliness = 1.0 - (forbidden_alive / max(1, len(removed)))
    ent_retention = len(alive_ents & actual_entities) / max(1, len(alive_ents))

    score = 100.0 * (0.4 * retention + 0.3 * cleanliness + 0.3 * ent_retention)
    return round(max(0.0, min(100.0, score)), 1)


def retrieval_accuracy(retrieved_ids: Iterable[str], gold_ids: Iterable[str]) -> float:
    """Top-1 retrieval accuracy: did the retriever pick at least one gold doc?"""
    retrieved = list(retrieved_ids)
    gold = set(gold_ids)
    if not retrieved or not gold:
        return 0.0
    return 100.0 * (1.0 if any(r in gold for r in retrieved) else 0.0)
