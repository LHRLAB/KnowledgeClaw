"""Item scores in [0, 1].

  mc (T2)                  exact choice
  mc, two phases (T5)      0.5 per phase (before / after the increment)
  candidate set (T6)       balanced accuracy over the candidates
  free-form set (T1/T3/T4) one-to-one matching F1: a predicted value matches a gold value when the gold is a
                           date (YYYY, YYYY-MM, YYYY-MM-DD) and the prediction parses to it at the gold's
                           granularity, or their word-token F1 is >= TOKEN_F1, or they are equal after norm()
"""
from __future__ import annotations

import re
from datetime import datetime

TOKEN_F1 = 0.5
ART = re.compile(r"^(the|a|an|this|that|these|those|some|our|your|their)\s+")
DATE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")
WORD = re.compile(r"\w+", re.UNICODE)
_SENTINEL = [datetime(1111, 1, 1), datetime(2222, 2, 2)]      # two defaults tell a parsed field from a filled one


def norm(x) -> str:
    x = ART.sub("", str(x).lower().strip(" .,\"'"))
    return re.sub(r"\s+", " ", x)


def _cand_key(x) -> str:
    return norm(str(x).split("|", 1)[0])


def candidate_balanced_accuracy(item: dict, pred) -> float:
    """Per candidate "is it in the answer": accuracy on the positives and on the negatives, averaged."""
    cands = {_cand_key(c) for c in item["question"]["candidates"]}
    gold = {_cand_key(a) for a in item["answer"]} & cands
    got = {_cand_key(p) for p in (pred or [])} & cands
    pos, neg = gold, cands - gold
    parts = []
    if pos:
        parts.append(len(got & pos) / len(pos))
    if neg:
        parts.append(len(neg - got) / len(neg))
    return sum(parts) / len(parts) if parts else 1.0


def score_item(item: dict, pred) -> float:
    t = item["answer_type"]
    if pred is None:
        return 0.0
    try:
        if t == "mc":
            return float(int(pred) == int(item["gold_index"]))
        if t == "set" and isinstance(item.get("question", {}).get("candidates"), list):
            return candidate_balanced_accuracy(item, pred)
    except (TypeError, ValueError):
        return 0.0
    raise ValueError(f"unknown answer_type {t}")


def _parse_date(s: str):
    """(year, month|None, day|None) written in s, or None."""
    s = str(s).strip()
    m = DATE.match(s)
    if m:
        y, mo, d = m.groups()
        return int(y), int(mo) if mo else None, int(d) if d else None
    try:
        from dateutil import parser
        a, b = (parser.parse(s, default=d, fuzzy=True) for d in _SENTINEL)
    except Exception:
        return None
    y = a.year if a.year == b.year else None
    if y is None:
        return None
    return y, (a.month if a.month == b.month else None), (a.day if a.day == b.day else None)


def _sim(p: str, g: str) -> float:
    if DATE.match(str(g).strip()):
        pd = _parse_date(p)
        if not pd:
            return 0.0
        gy, gm, gd = _parse_date(g)
        ok = pd[0] == gy and (gm is None or pd[1] == gm) and (gd is None or pd[2] == gd)
        return 1.0 if ok else 0.0
    if norm(p) == norm(g):
        return 1.0
    pt, gt = set(WORD.findall(str(p).lower())), set(WORD.findall(str(g).lower()))
    if not pt or not gt:
        return 0.0
    return 2 * len(pt & gt) / (len(pt) + len(gt))


def soft_f1(pred: list, gold: list) -> float:
    pred = [p for p in pred if str(p).strip()]
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    pairs = sorted(((_sim(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)), reverse=True)
    used_p, used_g, tp = set(), set(), 0
    for s, i, j in pairs:
        if s < TOKEN_F1:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        tp += 1
    return 2 * tp / (len(pred) + len(gold))


def is_phased(item: dict) -> bool:
    return isinstance(item.get("gold_index"), dict)


def score(item: dict, pred) -> float:
    if is_phased(item):
        g = item["gold_index"]
        pred = pred or {}
        return 0.5 * (pred.get("A") == g.get("A")) + 0.5 * (pred.get("B") == g.get("B"))
    q = item.get("question") or {}
    if item["answer_type"] in ("set", "entity_set") and not isinstance(q.get("candidates"), list):
        if pred is None:
            return 0.0
        pred = pred if isinstance(pred, list) else [pred]
        return soft_f1([str(x) for x in pred], [str(x) for x in item["answer"]])
    return score_item(item, pred)
