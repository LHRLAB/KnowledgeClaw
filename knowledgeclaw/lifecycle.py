"""
KnowledgeClawAgent — closed-loop lifecycle orchestrator (C2).

The lifecycle agent is the public face of the contribution C2 from PAPER.md.
It exposes four explicit stages:

  1. CONSTRUCTION       — heterogeneous data → typed hyperedges
  2. QUERYING           — multi-granular hierarchical retrieval
  3. REASONING          — multi-step inferential traversal (ReACT)
  4. DYNAMIC UPDATE     — incremental ingestion + deletion + consistency

…and orchestrates them in a closed loop. Every Update transition runs a
consistency check and invalidates the community cache so that subsequent
Query / Reason calls see a coherent hypergraph. Lifecycle events are
recorded in a JSONL log under the storage directory so the user can audit
exactly what the agent did.

The lifecycle agent is intentionally a thin orchestration layer over the
existing engine — its value is not in re-implementing functionality but in
*declaring* the explicit lifecycle that distinguishes KnowledgeClaw from
flat memory systems like MemGPT or Mem0.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


class KnowledgeClawAgent:
    """Closed-loop orchestrator for the four lifecycle stages."""

    STAGES = ("construction", "querying", "reasoning", "update")

    def __init__(self, engine):
        self._engine = engine
        self._log_path = Path(engine._storage_dir) / "lifecycle.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------- public stages

    def construct(self, folder_or_file: str) -> Dict[str, Any]:
        return self._stage("construction", lambda: self._engine.ingest(folder_or_file),
                           {"path": folder_or_file})

    def query(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        return self._stage("querying", lambda: self._engine.query(question, top_k=top_k),
                           {"question": question, "top_k": top_k})

    def reason(self, topic: str, max_rounds: int = 5) -> Dict[str, Any]:
        return self._stage("reasoning", lambda: self._engine.infer(topic, max_rounds=max_rounds),
                           {"topic": topic, "max_rounds": max_rounds})

    def update(self, file_path: str, force: bool = False) -> Dict[str, Any]:
        """Add a new file to the existing hypergraph and run a consistency check."""
        def _do():
            add_report = self._engine.update_file(file_path, force=force)
            consistency = self._engine.consistency_check()
            return {"add": add_report, "consistency": consistency}
        return self._stage("update", _do, {"file_path": file_path, "force": force})

    def delete(self, source_name: str) -> Dict[str, Any]:
        """Delete a source from the hypergraph and run a consistency check."""
        def _do():
            del_report = self._engine.delete_source(source_name)
            consistency = self._engine.consistency_check()
            return {"delete": del_report, "consistency": consistency}
        return self._stage("update", _do, {"delete_source": source_name})

    def rebuild_communities(self) -> Dict[str, Any]:
        return self._stage("update", self._engine.build_communities, {"action": "rebuild_communities"})

    # ------------------------------------------------------- batch lifecycle

    def run_lifecycle(self, actions: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute a sequence of lifecycle actions.

        Each action dict has the form ``{"stage": <stage>, ...stage-specific args}``.
        Recognised stages: construction, update, delete, querying, reasoning,
        rebuild_communities. Returns a list of per-step results."""
        results: List[Dict[str, Any]] = []
        for action in actions:
            stage = action.get("stage")
            try:
                if stage == "construction":
                    res = self.construct(action["path"])
                elif stage == "update":
                    res = self.update(action["file_path"], force=action.get("force", False))
                elif stage == "delete":
                    res = self.delete(action["source"])
                elif stage == "querying":
                    res = self.query(action["question"], top_k=action.get("top_k", 10))
                elif stage == "reasoning":
                    res = self.reason(action["topic"], max_rounds=action.get("max_rounds", 5))
                elif stage == "rebuild_communities":
                    res = self.rebuild_communities()
                else:
                    res = {"error": f"unknown stage: {stage}"}
            except Exception as exc:
                logger.exception("Lifecycle stage %s failed", stage)
                res = {"error": str(exc)}
            results.append({"action": action, "result": res})
        return results

    # ----------------------------------------------------------- audit log

    def history(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not self._log_path.exists():
            return []
        lines = self._log_path.read_text(encoding="utf-8").strip().splitlines()
        events = []
        for line in lines[-limit:]:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return events

    # --------------------------------------------------------- internal

    def _stage(self, stage: str, fn, params: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        try:
            result = fn() or {}
            ok = True
            error = None
        except Exception as exc:
            logger.exception("Lifecycle stage %s failed", stage)
            result = {}
            ok = False
            error = str(exc)
        elapsed = time.time() - start
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "params": params,
            "ok": ok,
            "elapsed_s": round(elapsed, 3),
            "error": error,
        }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return {"stage": stage, "ok": ok, "elapsed_s": round(elapsed, 3),
                "error": error, "result": result}
