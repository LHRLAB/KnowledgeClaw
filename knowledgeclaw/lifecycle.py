"""C2, the closed-loop lifecycle: Add, Del and Refresh keep K provenance-sound (every node's provenance is a
non-empty set of files still present, every index entry points to a live node).

  Add(D)          perceive -> extract -> merge. Identity rules make it a merge, not an append.
  Del(d)          a node owned by d alone is removed with its index entries; a shared node loses d from its
                  provenance; a garbage-collection pass drops what that orphaned.
  Refresh(d, d')  Del(d) then Add(d'), and d's handle is re-pointed at d', so a question that names d is
                  answered from the new version and no path reaches the old one.
Construction is Add folded over the corpus, so after any edit sequence a node exists iff some file still
present supports it. Every edit is appended to an audit log; derived structures (cross-modal hyperedges,
community reports) are recomputed or invalidated by every edit.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from knowledgeclaw import extract as X
from knowledgeclaw.hypergraph import FileNode, Hypergraph, hid, norm
from knowledgeclaw.perceive import Perceived, Source, perceive

WORKERS = 24


def pmap(fn, xs, workers=WORKERS):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, xs))


class Lifecycle:
    def __init__(self, kg: Hypergraph, on_change=None, structured: bool = True):
        self.kg = kg
        self.structured = structured        # False = ablation w/o C1: segments of the text projection only
        self.log: list[dict] = []
        self.on_change = on_change or (lambda: None)

    def _audit(self, op: str, params: dict, report: dict, t0: float) -> dict:
        ev = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "op": op, "params": params, "report": report,
              "elapsed_s": round(time.time() - t0, 2)}
        self.log.append(ev)
        return ev

    # ------------------------------------------------------------------------------------------ Add
    def prepare(self, sources: list[Source]) -> list[tuple]:
        """The model work of Add (perception and extraction), parallel and side-effect free."""
        llm = self.kg.llm
        if not self.structured:
            return [(perceive(Source(s.name, s.modality, s.text, None, s.revises), llm), {}) for s in sources]
        seen = pmap(lambda s: perceive(s, llm), sources)
        recs = pmap(lambda p: X.record("\n".join(p.segments), llm), seen)
        return [(p, r) for p, r in zip(seen, recs)]

    def add(self, sources: list[Source], prepared: list[tuple] | None = None, audit: bool = True) -> dict:
        t0 = time.time()
        prepared = prepared if prepared is not None else self.prepare(sources)
        seen = [p for p, _ in prepared]
        touched: set = set()
        new_edges, new_ents = [], []
        for p, rec in prepared:
            e, t = self._write(p, rec)
            new_edges += e
            new_ents += t
            touched |= set(t)
        self._index(seen, new_ents, new_edges)
        self._derive_xmodal(touched)
        self.on_change()
        rep = {"files": len(seen), "native_media": sum(p.native for p in seen),
               "hyperedges": len(set(new_edges)), "entities": len(set(new_ents))}
        return self._audit("add", {"files": [s.name for s in sources]}, rep, t0) if audit else rep

    def _write(self, p: Perceived, rec: dict) -> tuple[list[str], list[str]]:
        kg, src = self.kg, p.source
        f, m = src.name, src.modality
        with kg.lock:
            kg.files[f] = FileNode(f, m, p.segments, p.description, p.native, rec, src.revises)
            kg.version += 1
        edges, ents = [], []
        fields = dict(rec.get("fields") or {})
        if rec.get("title"):
            fields = {"title": [rec["title"]], **fields}
        for rel, vals in fields.items():
            ids = [kg.put_entity(v, rel, f) for v in vals]
            ents += ids
            edges.append(kg.put_edge("record", rel, f"{f} | {rel}: " + "; ".join(vals), ids, [f], m, f,
                                     key=hid("record", f, rel)))
        for ev in rec.get("events") or []:
            edges.append(kg.put_edge("event", ev["event"], f"{f} | {ev['event']}: {ev['date']}", [], [f], m, f,
                                     key=hid("event", f, ev["event"], ev["date"])))
        return edges, ents

    def _index(self, seen: list[Perceived], ents: list[str], edges: list[str]) -> None:
        kg = self.kg
        ents = list(dict.fromkeys(ents))
        edges = list(dict.fromkeys(edges))
        ents = [e for e in ents if e in kg.entities]
        kg.I_V.add(ents, [kg.entities[e].name for e in ents])
        kg.I_E.add(edges, [kg.edges[h].text for h in edges])
        keys, texts = [], []
        for p in seen:
            for j, s in enumerate(p.segments):
                keys.append(f"{p.source.name}#{j}")
                texts.append(s)
        kg.I_S.add(keys, texts)

    def _derive_xmodal(self, eids) -> None:
        """One cross-modal hyperedge per entity read from files of >= 2 modalities (anywhere in K)."""
        kg = self.kg
        for e in eids:
            if e not in kg.entities:
                continue
            files = sorted(f for f in kg.entity_files(e) if f in kg.files)
            mods = sorted({kg.files[f].modality for f in files})
            key = hid("xmodal", e)
            old = kg.edges.get(key)
            if old is not None:
                self._remove_edge(key)
            if len(mods) >= 2:
                kg.put_edge("xmodal", "cross-modal", f"{kg.entities[e].name} | read from {', '.join(mods)}",
                            [e], files, "+".join(mods), files[0], key=key)
                kg.edges[key].prov = set(files)

    # ------------------------------------------------------------------------------------------ Del
    def _remove_edge(self, h: str) -> None:
        kg = self.kg
        x = kg.edges.pop(h, None)
        if x is None:
            return
        for m in x.members:
            kg.ent_edges.get(m, set()).discard(h)
        for f in x.files:
            kg.file_edges.get(f, set()).discard(h)
        kg.I_E.drop([h])

    def delete(self, handle: str, audit: bool = True) -> dict:
        """Del of a handle removes every file currently holding it (all files of its current version)."""
        t0 = time.time()
        reps = [self._delete_file(f) for f in (self.kg.current(handle) or [handle])]
        self.on_change()
        rep = {k: sum(r[k] for r in reps) for k in reps[0] if k != "file"} if reps else {}
        rep["files"] = [r["file"] for r in reps]
        if audit:
            self._audit("del", {"file": handle}, rep, t0)
        return rep

    def _delete_file(self, f: str) -> dict:
        kg = self.kg
        rep = {"file": f, "hyperedges_removed": 0, "hyperedges_pruned": 0, "entities_removed": 0,
               "entities_pruned": 0, "segments_removed": 0}
        with kg.lock:
            node = kg.files.pop(f, None)
            kg.version += 1
            if node is not None:
                rep["segments_removed"] = kg.I_S.drop([f"{f}#{j}" for j in range(len(node.segments))])
            touched = set()
            for h in list(kg.edges):                                   # Eq. (4) over hyperedges
                x = kg.edges[h]
                if f not in x.prov and f not in x.files:
                    continue
                touched |= set(x.members)
                if x.prov <= {f}:
                    self._remove_edge(h)
                    rep["hyperedges_removed"] += 1
                else:
                    x.prov.discard(f)
                    if f in x.files:
                        x.files.remove(f)
                    rep["hyperedges_pruned"] += 1
            kg.file_edges.pop(f, None)
            for e in list(kg.entities):                                # Eq. (4) over entities
                ent = kg.entities[e]
                if f not in ent.prov:
                    continue
                touched.add(e)
                if ent.prov <= {f}:
                    self._remove_entity(e)
                    rep["entities_removed"] += 1
                else:
                    ent.prov.discard(f)
                    rep["entities_pruned"] += 1
            rep.update(self.gc())
            self._derive_xmodal(touched)
        return rep

    def _remove_entity(self, e: str) -> None:
        kg = self.kg
        kg.entities.pop(e, None)
        kg.ent_edges.pop(e, None)
        for s in kg.pkey.values():
            s.discard(e)
        kg.I_V.drop([e])

    def gc(self) -> dict:
        """Drop what a removal orphaned: hyperedges left with no member and no file, entities with no hyperedge."""
        kg, rep = self.kg, {"gc_hyperedges": 0, "gc_entities": 0}
        for h in list(kg.edges):
            x = kg.edges[h]
            x.members = [m for m in x.members if m in kg.entities]
            x.files = [f for f in x.files if f in kg.files]
            if not x.files and not x.members or x.kind in ("record", "event") and not x.files:
                self._remove_edge(h)
                rep["gc_hyperedges"] += 1
        for e in list(kg.entities):
            if not any(h in kg.edges for h in kg.ent_edges.get(e, ())):
                self._remove_entity(e)
                rep["gc_entities"] += 1
        return rep

    # ------------------------------------------------------------------------------------------ Refresh
    def refresh(self, handle: str, new: list[Source], prepared: list[tuple] | None = None) -> dict:
        """Refresh(d -> d'): the new version (one or more files, e.g. a PDF and its text) replaces every file
        currently holding the handle; the handle and the superseded files are re-pointed at the new version."""
        t0 = time.time()
        old = self.kg.current(handle) or []
        for s in new:
            s.revises = handle
        d = self.delete(handle, audit=False)
        a = self.add(new, prepared, audit=False)
        with self.kg.lock:
            for h in {handle, *old}:
                self.kg.alias[h] = [s.name for s in new]
        return self._audit("refresh", {"file": handle, "old": old, "new": [s.name for s in new]},
                           {"del": d, "add": a}, t0)

    # ------------------------------------------------------------------------------------------ invariant
    def check(self) -> dict:
        """Provenance soundness, Eq. (3): every provenance non-empty and live, every index key a live node."""
        kg = self.kg
        live = set(kg.files)
        bad_e = [e for e, x in kg.entities.items() if not x.prov or not x.prov <= live]
        bad_h = [h for h, x in kg.edges.items() if not x.prov or not x.prov <= live]
        bad_i = [k for k in kg.I_V.row if k not in kg.entities] + [k for k in kg.I_E.row if k not in kg.edges] + \
            [k for k in kg.I_S.row if k.rsplit("#", 1)[0] not in live]
        return {"sound": not (bad_e or bad_h or bad_i), "bad_entities": len(bad_e), "bad_hyperedges": len(bad_h),
                "bad_index": len(bad_i)}
