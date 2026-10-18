"""The shared answer step p_ans: question + evidence -> the JSON answer contract -> a scorable prediction.

Every compared method answers through this same prompt, so methods differ only in the evidence they hand in.
"""
from __future__ import annotations

import json

SYS_CB = ("You are taking a closed-book exam. You have NO access to any document, image, audio or video. "
          "Answer from the question text alone; guess if you must.")
SYS_R = ("You may use the CONTEXT below (retrieved automatically over the released corpus; it may be "
         "irrelevant, incomplete, or -- in the ablation condition -- deliberately missing one modality). "
         "Do not assume anything not stated in the context or the question.")

PLACEHOLDER_VALUES = {"unknown", "n/a", "na", "none", "null", "not applicable", "not specified", "n\\a", ""}


def cand_list(it: dict) -> list | None:
    c = (it.get("question") or {}).get("candidates")
    return c if isinstance(c, list) else None


def opt_text(o) -> str:
    return o if isinstance(o, str) else json.dumps(o, sort_keys=True, ensure_ascii=False)


def stem_text(it: dict) -> str:
    q = dict(it.get("question") or {})
    for k in ("candidates", "options"):
        q.pop(k, None)
    out = []

    def walk(x):
        if isinstance(x, dict):
            for k in sorted(x):
                walk(x[k])
        elif isinstance(x, list):
            for v in x:
                walk(v)
        elif isinstance(x, (str, int, float)) and not isinstance(x, bool):
            out.append(str(x))
    walk(q)
    return " ".join(out)


def build_prompt(it: dict, ctx_text: str | None, phase: str | None = None) -> tuple:
    """phase: None for single-phase items; "A" or "B" for T5 (answer, commit the increment, answer again)."""
    q = it["question"]["text"] if isinstance(it.get("question"), dict) and "text" in it["question"] \
        else stem_text(it)
    sysmsg = SYS_CB if ctx_text is None else SYS_R
    if phase == "B" and ctx_text is not None:
        q = f"(An update to the underlying record may have been filed since any earlier question.) {q}"
    body = f"CONTEXT:\n{ctx_text}\n\nQUESTION: {q}" if ctx_text is not None else f"QUESTION: {q}"
    if it["answer_type"] == "mc":
        opts = "\n".join(f"[{k}] {opt_text(o)}" for k, o in enumerate(it["options"], 1))
        body += f"\n\nOptions:\n{opts}\n\nRespond with JSON only: {{\"choice\": <1-{len(it['options'])}>}}"
    elif (it["answer_type"] in ("set", "entity_set") and cand_list(it) is None
          and "entity_types" in (it.get("question") or {})):
        body += ("\n\nRespond with JSON only: {\"entities\": [\"<entity name>\", \"<entity name>\", ...]}. "
                 "Each element must be ONLY the entity's own name/surface form, exactly as it appears -- "
                 "for example a person's name, an organisation, a subject term, or a physical part or "
                 "feature. Do NOT list the entity TYPES you were asked about (e.g. do not answer with "
                 "\"author\" or \"subject\" themselves), do NOT wrap entities in objects with their own "
                 "type/modality fields, and do NOT answer \"unknown\" -- if you cannot find any, respond "
                 "with an empty list.")
    elif it["answer_type"] in ("set", "entity_set") and cand_list(it) is None:
        body += ("\n\nAnswer the question directly. Respond with JSON only: "
                 "{\"answer\": [\"<value>\"]} -- a list containing the specific value(s) the question asks "
                 "for (usually exactly one, e.g. a single date, name or identifier), not a list of "
                 "unrelated entities found in the context.")
    elif it["answer_type"] == "set" and cand_list(it) is not None:
        cands = "\n".join(f"- {c}" for c in cand_list(it))
        body += (f"\n\nCandidates:\n{cands}\n\nRespond with JSON only: "
                 f"{{\"covered\": [\"<candidate>\", ...]}}. Each element must be one of the candidate "
                 f"strings above, copied exactly, as a plain string (not wrapped in an object).")
    else:
        body += "\n\nRespond with JSON only: {\"answer\": \"...\"}"
    return sysmsg, body


def _clean_str_list(xs: list) -> list:
    """Unwrap {"name": ...} objects to their name and drop placeholder values."""
    out = []
    for x in xs:
        v = x
        if isinstance(x, dict):
            v = x.get("name") or x.get("entity") or x.get("value")
        if v is None:
            continue
        v = str(v).strip()
        if not v or v.lower() in PLACEHOLDER_VALUES:
            continue
        out.append(v)
    return out


def pred_from_response(it: dict, r) -> object:
    if not isinstance(r, dict):
        return None
    if it["answer_type"] == "mc":
        c = r.get("choice")
        try:
            c = int(c)
        except (TypeError, ValueError):
            return None
        n = len(it.get("options") or [])
        return c - 1 if 1 <= c <= n else None
    if (it["answer_type"] in ("set", "entity_set") and cand_list(it) is None
            and "entity_types" in (it.get("question") or {})):
        e = r.get("entities")
        return _clean_str_list(e) if isinstance(e, list) else None
    if it["answer_type"] in ("set", "entity_set") and cand_list(it) is None:
        a = r.get("answer")
        if isinstance(a, list):
            return _clean_str_list(a)
        if isinstance(a, (str, int, float)):
            return _clean_str_list([a])
        return None
    if it["answer_type"] == "set" and cand_list(it) is not None:
        c = r.get("covered")
        return _clean_str_list(c) if isinstance(c, list) else None
    return r.get("answer")
