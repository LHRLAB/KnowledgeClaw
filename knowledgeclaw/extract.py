"""C1, extraction: the model turns what was read into hyperedges.

One call over the file's opening segments (or its media description) returns the file's catalogue record:
typed fields (author, affiliation, subject, ...) and dated events. Every field becomes one n-ary record
hyperedge joining the file to all of that field's values; every event becomes one event hyperedge.
"""
from __future__ import annotations

import re

from knowledgeclaw.backend import Backend

RECORD_CHARS = 9000
FIELDS = ("author", "affiliation", "subject", "keyword", "identifier", "organization", "person", "place",
          "depicts", "object_type", "element", "language", "venue")
DATE = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")

RECORD_PROMPT = """You are cataloguing one file of a personal knowledge base. From the file content below, \
write its catalogue record. Copy names, identifiers and dates exactly as they appear; never invent a value \
that the content does not support.

Return JSON only:
{{"title": "<title as printed, or a short descriptive title>",
  "kind": "<what the file is: preprint, journal article, report, bill, thesis, abstract, photograph, \
photomicrograph, drawing, floor plan, lecture recording, interview, song, ...>",
  "summary": "<one sentence>",
  "fields": {{
    "author": ["<each author's full name, in order>"],
    "affiliation": ["<each distinct affiliation, the full string as printed>"],
    "subject": ["<the controlled-vocabulary subject terms a library catalogue would assign to this file, each \
in the vocabulary's exact preferred form: arXiv category codes (e.g. cs.CL, eess.AS, q-bio.NC) for arXiv \
preprints; 5-15 MeSH descriptors for biomedical articles, written as MeSH writes them (plural nouns, \
inverted forms with commas), covering the population studied, the conditions, organisms, interventions, \
settings and methods, plus check tags (Humans, Male, Female, age groups) when the study population supports \
them; policy areas for government reports; subject headings otherwise>"],
    "keyword": ["<author keywords or key topics>"],
    "identifier": ["<DOI, arXiv id, report or bill number, PMID, ...>"],
    "venue": ["<journal, conference, publisher or series>"],
    "organization": ["<other organisations named>"],
    "person": ["<other people named: speakers, editors, people shown>"],
    "place": ["<places>"],
    "depicts": ["<for media: what is shown or heard: people, organisms, objects, activities>"],
    "object_type": ["<for media: the kind of object, e.g. Photographs, Portraits, Engravings, Maps>"],
    "element": ["<for plans, diagrams and building photos: every room type and building element shown, each \
as a canonical singular class name (e.g. Bedroom, Kitchen, Door, Window, Wall, Stairs, Balcony, Roof)>"],
    "language": ["<language of the content>"]
  }},
  "events": [{{"event": "<submitted | received | revised | accepted | published | published online | \
published print | created | recorded | issued | ...>", "date": "<YYYY-MM-DD, YYYY-MM or YYYY>"}}]}}

Rules: leave a field as [] when the content does not state it. An arXiv stamp such as \
"arXiv:2205.01234v2 [cs.CL] 24 May 2022" gives the identifier, the primary subject category and the \
submission date of that version. Journal front matter gives received / accepted / published dates.

FILE:
{content}"""

def _strs(xs) -> list[str]:
    out = []
    for x in xs if isinstance(xs, list) else ([xs] if isinstance(xs, str) else []):
        if isinstance(x, dict):
            x = x.get("name") or x.get("value") or ""
        s = str(x).strip()
        if s and s.lower() not in {"unknown", "n/a", "none", "null"} and len(s) < 400:
            out.append(s)
    return list(dict.fromkeys(out))


def _date(s: str) -> str | None:
    s = str(s).strip()
    if DATE.match(s):
        return s
    try:
        from dateutil import parser
        return parser.parse(s, fuzzy=False).date().isoformat()
    except Exception:
        return None


def record(content: str, llm: Backend) -> dict:
    r = llm.chat([{"role": "user", "content": RECORD_PROMPT.format(content=content[:RECORD_CHARS])}],
                 json_mode=True, max_tokens=2500)
    r = r if isinstance(r, dict) else {}
    f = r.get("fields") if isinstance(r.get("fields"), dict) else {}
    fields = {k: _strs(f.get(k)) for k in FIELDS}
    events = []
    for e in r.get("events") or []:
        if isinstance(e, dict) and e.get("event") and e.get("date"):
            d = _date(e["date"])
            if d:
                events.append({"event": str(e["event"]).strip().lower(), "date": d})
    return {"title": str(r.get("title") or "").strip(), "kind": str(r.get("kind") or "").strip(),
            "summary": str(r.get("summary") or "").strip(), "fields": {k: v for k, v in fields.items() if v},
            "events": events}
