"""KnowledgeClaw: the agent that builds, maintains and answers through an omnimodal typed hypergraph.

    kc = KnowledgeClaw(backend)
    kc.construct(sources)             # C1: perceive -> extract -> merge (Add folded over the corpus)
    kc.commit([new_version])          # C2: Refresh when the source revises a file, Add otherwise
    kc.delete("old.pdf")              # C2: Del with provenance pruning + garbage collection
    ev = kc.retrieve("question ...")  # C3: routed evidence (subgraph / community / hybrid)
    kc.dump(out_dir)                  # the artefact, the audit log and the invariant check

Construction, update and query share one loop: what an update commits is what the next query sees. Every
edit is recorded in the audit log.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np

from knowledgeclaw import community as C
from knowledgeclaw import retrieve as R
from knowledgeclaw import router
from knowledgeclaw.backend import Backend
from knowledgeclaw.hypergraph import Hypergraph
from knowledgeclaw.lifecycle import Lifecycle
from knowledgeclaw.perceive import Source


class KnowledgeClaw:
    def __init__(self, llm: Backend):
        self.llm = llm
        self.kg = Hypergraph(llm)
        self._reports: list[dict] | None = None
        self._lock = threading.Lock()
        self.lifecycle = Lifecycle(self.kg, on_change=self._invalidate)
        self._qcache: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------ construct / update (C1, C2)
    def construct(self, sources: list[Source]) -> dict:
        return self.lifecycle.add(sources)

    def commit(self, sources: list[Source]) -> list[dict]:
        """Fold an increment into K. Sources that revise a file are grouped by that file (a new version may
        span several files) and each group is one Refresh; the rest are Added. The model work of every group
        runs in parallel, the graph edits one after another."""
        groups: dict[str, list[Source]] = {}
        fresh: list[Source] = []
        for s in sources:
            if s.revises and self.kg.current(s.revises):
                groups.setdefault(s.revises, []).append(s)
            else:
                fresh.append(s)
        prepared = self.lifecycle.prepare(sources)
        by_name = {p.source.name: (p, r) for p, r in prepared}
        out = [self.lifecycle.refresh(h, ss, [by_name[s.name] for s in ss]) for h, ss in groups.items()]
        if fresh:
            out.append(self.lifecycle.add(fresh, [by_name[s.name] for s in fresh]))
        return out

    def delete(self, handle: str) -> dict:
        return self.lifecycle.delete(handle)

    def _invalidate(self) -> None:
        self._reports = None

    def communities(self) -> list[dict]:
        with self._lock:
            if self._reports is None:
                t0 = time.time()
                self._reports = C.build(self.kg)
                self.lifecycle._audit("communities", {}, {"communities": len(self._reports)}, t0)
            return self._reports

    # ------------------------------------------------------------------ query (C3)
    def _vec(self, text: str) -> np.ndarray:
        v = self._qcache.get(text)
        if v is None:
            v = np.asarray(self.llm.embed([text])[0], dtype=np.float32)
            v /= np.linalg.norm(v) + 1e-9
            self._qcache[text] = v
        return v

    def retrieve(self, question: str) -> dict:
        qv = self._vec(question)
        files, ents = R.anchors(self.kg, question, self._vec)
        intent = router.route(question, bool(files or ents), self.llm)
        out = {"intent": intent, "text": ""}
        if intent in ("subgraph", "hybrid"):
            out.update(R.subgraph(self.kg, question, qv, self._vec))
        if intent in ("community", "hybrid"):
            c = R.community(self.kg, self.communities(), qv)
            out["communities"] = c["communities"]
            out["text"] = "\n\n".join(t for t in (out["text"], c["text"]) if t)
        return out

    # ------------------------------------------------------------------ audit
    def dump(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        self.kg.dump(out / "hypergraph.json")
        (out / "audit.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n"
                                                  for e in self.lifecycle.log))
        if self._reports is not None:
            (out / "communities.json").write_text(json.dumps(self._reports, ensure_ascii=False, indent=1))
        (out / "invariant.json").write_text(json.dumps(self.lifecycle.check(), indent=1))
