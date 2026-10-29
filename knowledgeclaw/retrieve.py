"""C3, retrieval over K.

Subgraph tier, for a query q:
  1. anchors -- file handles named in q resolve (through Refresh aliases) to file nodes; quoted spans
     resolve to entity nodes, by identity norm() and else by their nearest entity in I_V;
  2. expansion -- files that hold an anchor entity, and files that share a catalogue entity with an anchor
     file, weighted by the entity's inverse file frequency (a rare shared author is a join, a common subject
     term barely one); conjunctive matches rank first;
  3. dense paths -- the entity path (top-2k entities of I_V, walked to their hyperedges) and the relation
     path (top-2k hyperedges of I_E) fused by reciprocal rank, plus the top segments of I_S.
The evidence is the anchor files' full cards (record, events, native description, opening segments), the
linked files' cards with the entity each shares, and the fused hyperedges and segments.
Community tier: the top reports of I_C.
"""
from __future__ import annotations

import re

import numpy as np

from knowledgeclaw.extract import FIELDS
from knowledgeclaw.hypergraph import FileNode, Hypergraph, norm

HANDLE = re.compile(r"\b[0-9a-f]{16}\.[A-Za-z0-9]{2,4}\b")
QUOTED = re.compile(r"[\"\u201c\u201d]([^\"\u201c\u201d]{2,300})[\"\u201c\u201d]|(?<![\w])'([^']{2,200}?)'(?![\w])")
K = 10                      # fused hyperedges kept
LAMBDA = 4                  # RRF constant
MAX_LINKED = 6              # linked works kept
KEEP_RATIO = 0.6            # a linked work is kept within this ratio of the best score
ENTITY_MATCH = 0.86         # cosine above which a quoted span is taken to name an existing entity
ANCHOR_CHARS = 7000
LINKED_CHARS = 700
BUDGET = 22000


LINKED_FIELDS = ("author", "subject", "keyword", "identifier", "venue", "depicts", "object_type")


def card(kg: Hypergraph, f: FileNode, handle: str | None = None, body_chars: int = 0, note: str = "",
         lines: tuple | None = None) -> str:
    r = f.record or {}
    head = f"=== {f.modality} file {handle or f.name}"
    if handle and handle != f.name:
        head += f" (current version: {f.name}, a new version of {handle})"
    head += f"{' -- ' + note if note else ''} ==="
    keep, lines = lines, [head]
    if r.get("title"):
        lines.append(f"title: {r['title']}")
    if r.get("kind"):
        lines.append(f"kind: {r['kind']}")
    for k, v in (r.get("fields") or {}).items():
        if keep is None or k in keep:
            lines.append(f"{k}: " + "; ".join(v))
    if r.get("events"):
        lines.append("events: " + "; ".join(f"{e['event']} {e['date']}" for e in r["events"]))
    if r.get("summary"):
        lines.append(f"summary: {r['summary']}")
    if body_chars:
        if f.description:
            lines.append(("native description: " if f.native else "description: ") + f.description[:body_chars])
        else:
            body = "\n".join(s.partition("\n")[2] for s in f.segments)
            lines.append("content: " + body[:body_chars])
    return "\n".join(lines)


def work_card(kg: Hypergraph, files: list[str], note: str, body_chars: int = 0) -> str:
    """One card for a work held in one or more files: the files' records merged, the first file's wording kept."""
    nodes = [kg.files[f] for f in files]
    head = "=== " + " = ".join(f"{n.modality} file {n.name}" for n in nodes)
    head += (" (the same work)" if len(nodes) > 1 else "") + (f" -- {note}" if note else "") + " ==="
    lines, fields, events = [head], {}, []
    rec0 = nodes[0].record or {}
    for k in ("title", "kind"):
        v = next((n.record.get(k) for n in nodes if (n.record or {}).get(k)), "")
        if v:
            lines.append(f"{k}: {v}")
    for n in nodes:
        for k, v in ((n.record or {}).get("fields") or {}).items():
            if k in LINKED_FIELDS:
                fields.setdefault(k, [])
                fields[k] += [x for x in v if x not in fields[k]]
        for e in (n.record or {}).get("events") or []:
            if e not in events:
                events.append(e)
    lines += [f"{k}: " + "; ".join(v) for k, v in fields.items()]
    if events:
        lines.append("events: " + "; ".join(f"{e['event']} {e['date']}" for e in events))
    if rec0.get("summary"):
        lines.append(f"summary: {rec0['summary']}")
    if body_chars:
        n = nodes[0]
        body = n.description or "\n".join(x.partition("\n")[2] for x in n.segments)
        lines.append("content: " + body[:body_chars])
    return "\n".join(lines)


def spans(q: str) -> list[str]:
    found = [a or b for a, b in QUOTED.findall(q)]
    return [x.strip() for x in dict.fromkeys(found) if x.strip() and not HANDLE.fullmatch(x.strip())]


def anchors(kg: Hypergraph, q: str, qvec_of) -> tuple[list[tuple[str, str]], list[str]]:
    files, ents = [], []
    for h in dict.fromkeys(HANDLE.findall(q)):
        for cur in kg.current(h):
            files.append((h, cur))
    for span in spans(q):
        e = norm(span)
        if e in TYPE_NAMES or e.replace(" ", "_") in TYPE_NAMES:      # a quoted field/type name is not an entity
            continue
        if e in kg.entities:
            ents.append(e)
            continue
        hits = kg.I_V.search_vec(qvec_of(span), 3)
        # a near neighbour counts only for a phrase; a code or a single token must match exactly
        if hits and hits[0][0] >= ENTITY_MATCH and len(re.findall(r"[a-z]{3,}", e)) >= 2:
            ents.append(hits[0][1])
    return files, list(dict.fromkeys(ents))


TYPE_NAMES = set(FIELDS) | {"title", "entity", "entities", "element_class", "room_type", "subject term"}
CATALOGUE = 10.0            # an anchor entity in a file's catalogue record
HEADING = 6.0              # a named term inside a file's title, subject or keyword field
MENTION = 3.0               # a named phrase found in the file's text
LEXICAL = 4.0               # max contribution of the keyword path per named span


def expand(kg: Hypergraph, files: list[tuple[str, str]], ents: list[str], named: list[str]) -> list[tuple[list, str]]:
    """Linked files, best first, each with a note naming what links it to the anchors.

    Three paths score a file: holding an anchor entity (in its catalogue record, or elsewhere in K), the
    keyword path (BM25 of each named span over the file's text), and sharing a catalogue entity with an
    anchor file, weighted by inverse file frequency."""
    anchor = {f for _, f in files}
    anchor |= {t for f in list(anchor) for t in kg.twins(f)}      # an anchor's twin is the same work, not "other"
    score: dict[str, float] = {}
    why: dict[str, list[str]] = {}

    def bump(g, w, note):
        if g in kg.files and g not in anchor:
            score[g] = score.get(g, 0.0) + w
            why.setdefault(g, []).append(note)

    for e in ents:
        same = kg.same_as(e)
        cat = {f for x in same for f in kg.entity_record_files(x)}
        for g in set().union(*(kg.entity_files(x) for x in same)):
            if g in cat:
                bump(g, CATALOGUE, f'catalogued under "{kg.entities[e].name}"')
            else:
                bump(g, MENTION, f'mentions "{kg.entities[e].name}"')
    for span in named:
        stem = norm(span).rstrip("s")
        for g in kg.files:                                        # the term named in a file's title/subject/keywords
            r = kg.files[g].record or {}
            heads = [r.get("title", "")] + [v for k in ("subject", "keyword") for v in (r.get("fields") or {}).get(k, [])]
            if len(stem) >= 4 and any(stem in norm(h) for h in heads):
                bump(g, HEADING, f'"{span}" in its title/subjects/keywords')
        hits = kg.phrase_files(span)                              # files whose text contains the phrase
        w = MENTION if len(hits) <= max(8, len(kg.files) // 20) else 1.0
        for g in hits:
            bump(g, w, f'mentions "{span}"')
        for g, s in kg.lexical(span, 8):                          # ranked keyword path
            bump(g, LEXICAL * s * 0.5, f'text matches "{span}"')
    for _, f in files:
        for e in kg.file_entities(f):
            fs = set().union(*(kg.entity_files(x) for x in kg.same_as(e)))
            if len(fs) > max(20, len(kg.files) // 5):            # hub terms join nothing
                continue
            w = kg.idf(e)
            rel = "/".join(sorted(kg.entities[e].types)) or "entity"
            for g in fs:
                bump(g, w, f'shares {rel} "{kg.entities[e].name}"')
    # a work held in several modalities (a PDF and its text) is one candidate: its files' scores add up
    works, seen = [], set()
    for g in sorted(score, key=lambda g: -score[g]):
        if g in seen:
            continue
        w = [g] + sorted(t for t in kg.twins(g) if t not in anchor)
        seen.update(w)
        works.append((w, sum(score.get(x, 0.0) for x in w),
                      ", ".join(dict.fromkeys(n for x in w for n in why.get(x, [])))[:300]))
    works.sort(key=lambda x: -x[1])
    top = works[0][1] if works else 0.0
    works = [x for x in works if x[1] >= KEEP_RATIO * top]           # weak links only distract the reader
    return [(w, note) for w, _, note in works[:MAX_LINKED]]


def dense(kg: Hypergraph, qv: np.ndarray, k: int = K) -> tuple[list[str], list[str]]:
    ent_hits = kg.I_V.search_vec(qv, 2 * k)
    L_V = []
    for _, e in ent_hits:
        for h in sorted(kg.ent_edges.get(e, ())):
            if h in kg.edges and h not in L_V:
                L_V.append(h)
    L_E = [h for _, h in kg.I_E.search_vec(qv, 2 * k) if h in kg.edges]
    score: dict[str, float] = {}
    for L in (L_V, L_E):
        for r, h in enumerate(L):
            score[h] = score.get(h, 0.0) + 1.0 / (LAMBDA + r)
    top = sorted(score, key=lambda h: -score[h])[:k]
    segs = [s for _, s in kg.I_S.search_vec(qv, 4)]
    return top, segs


def segment_text(kg: Hypergraph, key: str) -> str | None:
    f, _, j = key.rpartition("#")
    node = kg.files.get(f)
    if node is None or not j.isdigit() or int(j) >= len(node.segments):
        return None
    return node.segments[int(j)]


JOIN = re.compile(r"\b(other|another|share[sd]?|sharing|same|also|both|common|whose|together|related|linked)\b", re.I)


def subgraph(kg: Hypergraph, q: str, qv: np.ndarray, qvec_of) -> dict:
    files, ents = anchors(kg, q, qvec_of)
    # granularity: a question about the named file itself is answered from that file alone; a relational
    # question (or one anchored on entities) walks the hypergraph to the files it joins
    relational = bool(ents) or not files or bool(JOIN.search(q))
    linked = expand(kg, files, ents, spans(q)) if relational else []
    edges, segs = dense(kg, qv) if relational else ([], [])
    parts, used = [], 0
    for h, f in files:
        c = card(kg, kg.files[f], h, ANCHOR_CHARS)
        parts.append(c)
        used += len(c)
    if linked:
        parts.append("=== linked records, strongest link first (catalogued under a named term > text match > "
                     "shared entity) ===")
    for rank, (w, why) in enumerate(linked, 1):
        c = work_card(kg, w, f"link #{rank}: {why}", LINKED_CHARS if not files else 0)
        if used + len(c) > BUDGET:
            break
        parts.append(c)
        used += len(c)
    anchor_files = {f for _, f in files}
    # once the walk has found the joined works, dense hits count only inside them (elsewhere they distract);
    # with no work found, the dense paths are the fallback evidence
    walked = {f for w, _ in linked for f in w}
    keep = (lambda fs: bool(set(fs) & walked)) if walked else (lambda fs: True)
    related = [kg.edges[h].text for h in edges if not (set(kg.edges[h].files) & anchor_files) and keep(kg.edges[h].files)]
    if related:
        c = "=== related hyperedges ===\n" + "\n".join(f"- {t}" for t in related)
        parts.append(c)
        used += len(c)
    for key in segs:
        if key.rpartition("#")[0] in anchor_files or not keep([key.rpartition("#")[0]]):
            continue
        s = segment_text(kg, key)
        if s and used + len(s) <= BUDGET:
            parts.append("=== passage ===\n" + s[:3000])
            used += min(len(s), 3000)
    return {"anchors": [f for _, f in files], "entities": ents, "linked": [w for w, _ in linked],
            "relational": relational, "hyperedges": edges, "segments": segs, "parts": parts,
            "text": "\n\n".join(parts)}


def community(kg: Hypergraph, reports: list[dict], qv: np.ndarray, k: int = 3) -> dict:
    by = {r["id"]: r for r in reports}
    hits = [by[c] for _, c in kg.I_C.search_vec(qv, max(2, k)) if c in by]
    text = "\n\n".join(f"=== community report ({', '.join(r['modalities'])}) ===\n{r['report']}" for r in hits)
    return {"communities": [r["id"] for r in hits], "text": text}
