"""C1, the artefact K = (V, E, A, pi): entities, hyperedges, file (asset) nodes and a provenance map.

  file node   one per source file (every modality; media files are its assets): segments, record, and
              the native media description. Identity: the file's handle.
  entity      identity norm(name); carries the types it was extracted under and its provenance.
  hyperedge   kind "record" (a file, a catalogue field and all of that field's values), "event" (a file,
              an event and its date), or "xmodal" (one entity and the modalities it was read from, derived
              from provenance).
Provenance pi(x) is the set of file handles x was read from. Writing a node or hyperedge that already
exists unions pi; deleting a file prunes it (lifecycle.py).

Four dense indices: I_V (entity names), I_E (hyperedge texts), I_S (segments), I_C (community reports).
"""
from __future__ import annotations

import json
import math
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path

import numpy as np

from knowledgeclaw.backend import Backend

EMBED_BATCH = 64
EMBED_WORKERS = 16


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    s = re.sub(r"[\s ]+", " ", s)
    return s.strip(" .,;:\"'()[]{}")


PERSON_TYPES = {"author", "person"}


def person_key(name: str) -> str | None:
    """Surface-independent key of a person name: first initial + family name ("M. Langtry" = "Max Langtry")."""
    toks = re.findall(r"[^\W\d_]+", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower())
    if len(toks) < 2:
        return None
    return f"{toks[0][0]} {toks[-1]}"


def hid(*parts: str) -> str:
    return sha1("\x1f".join(norm(p) for p in parts).encode()).hexdigest()[:16]


@dataclass
class FileNode:
    name: str
    modality: str
    segments: list[str] = field(default_factory=list)
    description: str = ""
    native: bool = False
    record: dict = field(default_factory=dict)
    revises: str | None = None          # the handle this file is a new version of


@dataclass
class Entity:
    id: str
    name: str
    types: set = field(default_factory=set)
    prov: set = field(default_factory=set)


@dataclass
class Hyperedge:
    id: str
    kind: str                           # record | event | xmodal
    relation: str                       # field / event name / "cross-modal"
    text: str
    members: list = field(default_factory=list)     # entity ids
    files: list = field(default_factory=list)       # file handles it attaches to (record/event: the file)
    modality: str = ""
    prov: set = field(default_factory=set)


class DenseIndex:
    """Cosine index with tombstones; keys map to rows so an entry can be dropped with the node it indexes."""

    def __init__(self, llm: Backend):
        self.llm = llm
        self.lock = threading.Lock()
        self.vecs = np.zeros((0, 0), dtype=np.float32)
        self.keys: list[str] = []
        self.row: dict[str, int] = {}
        self.alive: list[bool] = []

    def add(self, keys: list[str], texts: list[str]) -> None:
        with self.lock:
            todo = [(k, t) for k, t in zip(keys, texts) if k not in self.row or not self.alive[self.row[k]]]
        if not todo:
            return
        texts = [t[:8000] or " " for _, t in todo]
        batches = [texts[i:i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
        with ThreadPoolExecutor(EMBED_WORKERS) as ex:                  # batches in parallel, order kept
            v = np.asarray([x for b in ex.map(self.llm.embed, batches) for x in b], dtype=np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
        with self.lock:
            start = len(self.keys)
            self.vecs = v if self.vecs.size == 0 else np.vstack([self.vecs, v])
            for i, (k, _) in enumerate(todo):
                self.keys.append(k)
                self.alive.append(True)
                self.row[k] = start + i

    def drop(self, keys) -> int:
        n = 0
        with self.lock:
            for k in keys:
                i = self.row.pop(k, None)
                if i is not None and self.alive[i]:
                    self.alive[i] = False
                    n += 1
        return n

    def search_vec(self, e: np.ndarray, k: int) -> list[tuple[float, str]]:
        with self.lock:
            if not self.keys:
                return []
            sims = self.vecs @ e
            sims[~np.asarray(self.alive)] = -np.inf
            top = np.argsort(-sims)[:k]
            return [(float(sims[i]), self.keys[i]) for i in top if np.isfinite(sims[i])]

    def __len__(self):
        return sum(self.alive)


class Hypergraph:
    def __init__(self, llm: Backend):
        self.llm = llm
        self.lock = threading.RLock()
        self.files: dict[str, FileNode] = {}
        self.entities: dict[str, Entity] = {}
        self.edges: dict[str, Hyperedge] = {}
        self.ent_edges: dict[str, set] = {}
        self.file_edges: dict[str, set] = {}
        self.alias: dict[str, list] = {}            # superseded handle -> the files of its current version
        self.pkey: dict[str, set] = {}              # person key -> person entities written under it
        self.version = 0                            # bumped by every edit; derived caches key on it
        self._bm25 = None
        self._flat = None
        self.I_V, self.I_E, self.I_S, self.I_C = (DenseIndex(llm) for _ in range(4))

    # ---------------------------------------------------------------- writes (merge = union provenance)
    def put_entity(self, name: str, typ: str, file: str) -> str:
        eid = norm(name)
        with self.lock:
            e = self.entities.get(eid)
            if e is None:
                e = self.entities[eid] = Entity(eid, name.strip())
                self.ent_edges[eid] = set()
            e.types.add(typ)
            e.prov.add(file)
            if typ in PERSON_TYPES and (k := person_key(name)):
                self.pkey.setdefault(k, set()).add(eid)
        return eid

    def same_as(self, eid: str) -> set:
        """The entity and every person entity that resolves to the same person key."""
        e = self.entities.get(eid)
        if e is None or not e.types & PERSON_TYPES:
            return {eid}
        k = person_key(e.name)
        return ({x for x in self.pkey.get(k, ()) if x in self.entities} | {eid}) if k else {eid}

    def put_edge(self, kind: str, relation: str, text: str, members: list[str], files: list[str],
                 modality: str, file: str, key: str | None = None) -> str:
        h = key or hid(kind, relation, text)
        with self.lock:
            x = self.edges.get(h)
            if x is None:
                x = self.edges[h] = Hyperedge(h, kind, relation, text, list(dict.fromkeys(members)),
                                              list(dict.fromkeys(files)), modality)
            else:
                x.members = list(dict.fromkeys(x.members + members))
                x.files = list(dict.fromkeys(x.files + files))
            x.prov.add(file)
            for m in x.members:
                self.ent_edges.setdefault(m, set()).add(h)
            for f in x.files:
                self.file_edges.setdefault(f, set()).add(h)
        return h

    # ---------------------------------------------------------------- reads
    def current(self, handle: str, _seen=None) -> list[str]:
        """The live files that hold a handle: the file itself, or the files of its latest version."""
        seen = _seen or set()
        if handle in seen:
            return []
        seen.add(handle)
        if handle in self.alias:
            return list(dict.fromkeys(f for h in self.alias[handle] for f in self.current(h, seen)))
        return [handle] if handle in self.files else []

    def file_entities(self, f: str) -> set:
        return {m for h in self.file_edges.get(f, ()) if h in self.edges and self.edges[h].kind == "record"
                for m in self.edges[h].members}

    def entity_files(self, eid: str) -> set:
        return {f for h in self.ent_edges.get(eid, ()) if h in self.edges for f in self.edges[h].files} | \
            {f for f in self.entities[eid].prov if f in self.files} if eid in self.entities else set()

    def entity_record_files(self, eid: str) -> set:
        """Files whose catalogue record holds the entity."""
        return {f for h in self.ent_edges.get(eid, ()) if h in self.edges and self.edges[h].kind == "record"
                for f in self.edges[h].files}

    def twins(self, f: str) -> set:
        """Cross-modal coreference of works: other files sharing f's title or a DOI/PMID-like identifier
        (a document held as both PDF and text, a recording and its transcript)."""
        out = set()
        for h in self.file_edges.get(f, ()):
            x = self.edges.get(h)
            if x is None or x.kind != "record" or x.relation not in ("title", "identifier"):
                continue
            for e in x.members:
                if x.relation == "identifier" and len(e) < 6:          # short ids (page numbers, e-numbers) join nothing
                    continue
                for g in self.entity_record_files(e):
                    if g != f and g in self.files and self.files[g].modality != self.files[f].modality:
                        out.add(g)
        return out

    def phrase_files(self, phrase: str) -> list[str]:
        """Files whose text or native description contains the phrase (case- and space-insensitive)."""
        ph = norm(phrase)
        if len(ph) < 3:
            return []
        with self.lock:
            if self._flat is None or self._flat[0] != self.version:
                self._flat = (self.version, {n: norm(" ".join(f.segments + [f.description]))
                                             for n, f in self.files.items()})
            flat = self._flat[1]
        pat = re.compile(r"(?<!\w)" + re.escape(ph) + r"(?!\w)")
        return [n for n, t in flat.items() if pat.search(t)]

    def lexical(self, query: str, k: int = 8) -> list[tuple[str, float]]:
        """The keyword path: BM25 of a query over each file's full text, scores scaled to (0, 1]."""
        from rank_bm25 import BM25Okapi
        with self.lock:
            if self._bm25 is None or self._bm25[0] != self.version:
                names = sorted(self.files)
                docs = [re.findall(r"\w+", "\n".join(self.files[n].segments + [self.files[n].description]).lower())
                        for n in names]
                self._bm25 = (self.version, names, BM25Okapi(docs or [[""]]))
            _, names, bm = self._bm25
        toks = re.findall(r"\w+", query.lower())
        if not toks or not names:
            return []
        sc = bm.get_scores(toks)
        top = sorted(range(len(names)), key=lambda i: -sc[i])[:k]
        m = max(sc[top[0]], 1e-9)
        return [(names[i], float(sc[i] / m)) for i in top if sc[i] > 0]

    def idf(self, eid: str) -> float:
        n = max(len(self.files), 1)
        return math.log(1 + n / max(len(self.entity_files(eid)), 1))

    def stats(self) -> dict:
        kinds = {}
        for x in self.edges.values():
            kinds[x.kind] = kinds.get(x.kind, 0) + 1
        return {"files": len(self.files), "entities": len(self.entities), "hyperedges": kinds,
                "native_media": sum(f.native for f in self.files.values()),
                "index": {"V": len(self.I_V), "E": len(self.I_E), "S": len(self.I_S), "C": len(self.I_C)}}

    # ---------------------------------------------------------------- persistence (the audited artefact)
    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            d = {"files": {k: vars(v) for k, v in self.files.items()},
                 "entities": {k: {**vars(v), "types": sorted(v.types), "prov": sorted(v.prov)}
                              for k, v in self.entities.items()},
                 "edges": {k: {**vars(v), "prov": sorted(v.prov)} for k, v in self.edges.items()},
                 "alias": self.alias, "stats": self.stats()}
        path.write_text(json.dumps(d, ensure_ascii=False))
