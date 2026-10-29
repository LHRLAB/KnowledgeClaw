"""KnowledgeClaw on OmniKE-Bench.

The adapter
  * turns bench documents into Sources (text/pdf: the text projection; image/audio/video: the raw file,
    read natively once at ingest),
  * maps a T5 increment ("[pdf file <new>, a new version of <old>]") to a Refresh of <old>,
  * answers with the shared answer prompt over the evidence KnowledgeClaw retrieves,
  * saves the artefact, the audit log and one retrieval trace per answer for inspection.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from evaluation import data as D
from evaluation import llm
from evaluation import prompt as P
from knowledgeclaw import KnowledgeClaw as KC
from knowledgeclaw import Source

REVISES = re.compile(r"a new version of ([^\]\s,]+)")


class KnowledgeClaw:
    def __init__(self, out: Path):
        self.out = out

    def _sources(self, docs) -> list[Source]:
        out = []
        for d in docs:
            m = REVISES.search(d.text.partition("\n")[0])
            path = D.media_path(self.persona, d.modality, d.file) if d.modality in ("image", "audio", "video") else None
            out.append(Source(d.file, d.modality, d.text, path, m.group(1) if m else None))
        return out

    def build(self, persona: str, docs) -> None:
        self.persona = persona
        self.out.mkdir(parents=True, exist_ok=True)
        self.trace_lock = threading.Lock()
        (self.out / "traces.jsonl").write_text("")
        self.kc = KC(llm.Backend())
        self.kc.construct(self._sources(docs))
        self.kc.dump(self.out)

    def commit(self, docs) -> None:
        self.kc.commit(self._sources(docs))
        self.kc.dump(self.out)

    def answer(self, q: dict, phase: str | None = None):
        ev = self.kc.retrieve(q["question"]["text"])
        sysmsg, user = P.build_prompt(q, ev["text"], phase=phase)
        r = llm.chat([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}], max_tokens=600)
        pred = P.pred_from_response(q, r)
        row = {"qid": q["qid"], "phase": phase, "intent": ev.get("intent"), "anchors": ev.get("anchors"),
               "entities": ev.get("entities"), "linked": ev.get("linked"), "communities": ev.get("communities"),
               "context_chars": len(ev["text"]), "context": ev["text"], "pred": pred}
        with self.trace_lock:
            with open(self.out / "traces.jsonl", "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return pred
