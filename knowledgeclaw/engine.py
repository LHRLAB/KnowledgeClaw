"""
KnowledgeClaw — main engine class.

Wires together every subsystem implementing the three contributions of
PAPER.md:

  C1 (Omnimodal typed hypergraph)
      → builder.py + ingestor.py + modes/*.py

  C2 (Closed-loop agentic lifecycle)
      → lifecycle.KnowledgeClawAgent
      → updater.Updater
      → agents.{QAAgent, InferenceAgent, ResearchAgent, EvalAgent}

  C3 (Hierarchical multi-granular retrieval with intent routing)
      → retriever.retrieve              (subgraph tier)
      → community.{detect, build, search}  (community tier)
      → router.IntentRouter              (intent dispatch)

The class itself is a thin "kernel": it owns the storage handles, the
clients, and the lifecycle agent — every higher-level operation goes through
either the lifecycle agent (for audited mutations) or the QA / Inference /
Research / Eval agents (for read-only operations).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class KnowledgeClaw:
    """High-level interface to the KnowledgeClaw knowledge engineering system.

    Usage::

        kc = KnowledgeClaw(kb_dir="./my_kb")
        kc.ingest("./my_documents")
        kc.query("What is X?")           # routed via intent router
        kc.update_file("./new_doc.pdf")  # incremental + consistency check
        kc.delete_source("old_doc.pdf")  # consistent deletion
        kc.build_communities()           # rebuild community summaries
        kc.lifecycle.run_lifecycle([     # auditable batch lifecycle
            {"stage": "construction", "path": "./docs"},
            {"stage": "querying", "question": "Summarise everything"},
        ])
    """

    def __init__(self, kb_dir: str, storage_dir: Optional[str] = None):
        import sys
        project_root = str(Path(__file__).parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        import config as cfg
        from knowledgeclaw.embedder import Embedder
        from knowledgeclaw.extractor import LLMExtractor

        self._cfg = cfg
        self._kb_dir = str(Path(kb_dir).resolve())

        if storage_dir is None:
            self._storage_dir = str(Path(self._kb_dir) / ".knowledgeclaw")
        else:
            self._storage_dir = str(Path(storage_dir).resolve())

        Path(self._storage_dir).mkdir(parents=True, exist_ok=True)
        logger.info("KnowledgeClaw storage at: %s", self._storage_dir)

        self._init_storage()

        # ── Embedder + LLM client (shared across components) ─────────────── #
        self._embedder = Embedder(
            api_key=cfg.OPENAI_API_KEY,
            base_url=cfg.OPENAI_BASE_URL,
            model=cfg.EMBED_MODEL,
            dim=cfg.EMBED_DIM,
        )
        self._llm_client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_BASE_URL)

        self._extractor = LLMExtractor(
            api_key=cfg.OPENAI_API_KEY,
            base_url=cfg.OPENAI_BASE_URL,
            model=cfg.LLM_MODEL,
        )

        self._build_subsystems()

    # -----------------------------------------------------------------------
    # Storage / subsystem initialisation
    # -----------------------------------------------------------------------

    def _init_storage(self):
        """(Re)initialise all storage objects from disk."""
        from knowledgeclaw.storage import JsonKVStorage, VectorStorage, GraphStorage
        import config as cfg
        sd = self._storage_dir
        self._kv = JsonKVStorage(str(Path(sd) / "chunks.json"))
        self._vdb_entity = VectorStorage(str(Path(sd) / "entity_vdb.json"), embedding_dim=cfg.EMBED_DIM)
        self._vdb_hyperedge = VectorStorage(str(Path(sd) / "hyperedge_vdb.json"), embedding_dim=cfg.EMBED_DIM)
        self._vdb_media = VectorStorage(str(Path(sd) / "media_vdb.json"), embedding_dim=cfg.EMBED_DIM)
        self._vdb_community = VectorStorage(str(Path(sd) / "community_vdb.json"), embedding_dim=cfg.EMBED_DIM)
        self._graph = GraphStorage(str(Path(sd) / "knowledge_graph.graphml"))

    def _build_subsystems(self):
        """Construct router, agents, updater, lifecycle agent."""
        from knowledgeclaw.router import IntentRouter
        from knowledgeclaw.updater import Updater
        from knowledgeclaw.lifecycle import KnowledgeClawAgent
        from knowledgeclaw.agents import QAAgent, InferenceAgent, ResearchAgent, EvalAgent
        import config as cfg

        self._router = IntentRouter(
            llm_client=self._llm_client, model=cfg.LLM_MODEL, use_llm=True
        )

        self._updater = Updater(
            storage_dir=self._storage_dir,
            graph=self._graph,
            kv=self._kv,
            vdb_entity=self._vdb_entity,
            vdb_hyperedge=self._vdb_hyperedge,
            vdb_media=self._vdb_media,
            vdb_community=self._vdb_community,
        )

        agent_kwargs = dict(
            client=self._llm_client,
            model=cfg.LLM_MODEL,
            embedder=self._embedder,
            kv=self._kv,
            vdb_entity=self._vdb_entity,
            vdb_hyperedge=self._vdb_hyperedge,
            graph=self._graph,
            vdb_community=self._vdb_community,
            community_reports_path=str(Path(self._storage_dir) / "community_reports.json"),
            router=self._router,
        )
        self._qa_agent = QAAgent(**agent_kwargs)
        self._inference_agent = InferenceAgent(**agent_kwargs)
        self._research_agent = ResearchAgent(**agent_kwargs)
        self._eval_agent = EvalAgent(
            client=self._llm_client,
            model=cfg.LLM_MODEL,
            kv=self._kv,
            graph=self._graph,
        )

        self._lifecycle = KnowledgeClawAgent(self)

    def _rebuild_agents(self):
        """Rebuild all stateful subsystems after a storage reset."""
        self._build_subsystems()

    # -----------------------------------------------------------------------
    # CONSTRUCTION
    # -----------------------------------------------------------------------

    def ingest(self, folder_or_file: str) -> Dict[str, Any]:
        """Ingest a folder or single file into the typed hypergraph."""
        from knowledgeclaw.builder import build_from_folder, build_from_file

        path = Path(folder_or_file)
        kwargs = dict(
            kb_dir=self._storage_dir,
            extractor=self._extractor,
            embedder=self._embedder,
            kv=self._kv,
            vdb_entity=self._vdb_entity,
            vdb_hyperedge=self._vdb_hyperedge,
            graph=self._graph,
            max_tokens=self._cfg.KB_CHUNK_SIZE,
            overlap=self._cfg.KB_CHUNK_OVERLAP,
            extract_batch_size=self._cfg.KB_EXTRACT_BATCH,
            vdb_media=self._vdb_media,
            media_embedder=self._embedder,
            expr_dir=self._cfg.EXPR_DIR,
        )
        if path.is_file():
            return build_from_file(file_path=str(path), **kwargs)
        if path.is_dir():
            return build_from_folder(folder_path=str(path), **kwargs)
        raise FileNotFoundError(f"Path not found: {folder_or_file}")

    # -----------------------------------------------------------------------
    # QUERY / REASONING / RESEARCH / EVAL
    # -----------------------------------------------------------------------

    def query(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        top_k = top_k or self._cfg.KB_TOP_K
        return self._qa_agent.run(question, top_k=top_k)

    def infer(self, topic: str, max_rounds: int = 5) -> Dict[str, Any]:
        return self._inference_agent.run(topic, max_rounds=max_rounds)

    def evaluate(self) -> Dict[str, Any]:
        return self._eval_agent.run()

    def research(self, topic: str) -> Dict[str, Any]:
        return self._research_agent.run(topic, kb_dir=self._storage_dir)

    # -----------------------------------------------------------------------
    # DYNAMIC UPDATE / CONSISTENCY (C2)
    # -----------------------------------------------------------------------

    def update_file(self, file_path: str, force: bool = False) -> Dict[str, Any]:
        return self._updater.add_file(
            file_path=file_path,
            extractor=self._extractor,
            embedder=self._embedder,
            media_embedder=self._embedder,
            max_tokens=self._cfg.KB_CHUNK_SIZE,
            overlap=self._cfg.KB_CHUNK_OVERLAP,
            extract_batch_size=self._cfg.KB_EXTRACT_BATCH,
            expr_dir=self._cfg.EXPR_DIR,
            force=force,
        )

    def delete_source(self, source_name: str) -> Dict[str, Any]:
        return self._updater.delete_source(source_name)

    def consistency_check(self) -> Dict[str, Any]:
        return self._updater.consistency_check()

    # -----------------------------------------------------------------------
    # COMMUNITY DETECTION (C3 coarse tier)
    # -----------------------------------------------------------------------

    def build_communities(self, min_size: int = 2, max_communities: int = 64) -> Dict[str, Any]:
        from knowledgeclaw.community import (
            build_community_reports,
            persist_community_reports,
            embed_community_reports,
        )
        reports = build_community_reports(
            graph=self._graph,
            llm_client=self._llm_client,
            model=self._cfg.LLM_MODEL,
            min_size=min_size,
            max_communities=max_communities,
        )
        report_path = str(Path(self._storage_dir) / "community_reports.json")
        persist_community_reports(reports, report_path)
        added = embed_community_reports(reports, self._embedder, self._vdb_community)
        return {
            "communities": len(reports),
            "embedded": added,
            "report_path": report_path,
        }

    def list_communities(self) -> List[Dict[str, Any]]:
        from knowledgeclaw.community import load_community_reports
        return load_community_reports(str(Path(self._storage_dir) / "community_reports.json"))

    # -----------------------------------------------------------------------
    # LIFECYCLE AGENT (C2 facade)
    # -----------------------------------------------------------------------

    @property
    def lifecycle(self):
        return self._lifecycle

    # -----------------------------------------------------------------------
    # Visualisation / introspection
    # -----------------------------------------------------------------------

    def graph_data(self) -> Dict[str, Any]:
        """Return graph as D3-compatible JSON for the web UI."""
        g = self._graph._graph
        degree_map = dict(g.degree())

        nodes = []
        for node_id, data in g.nodes(data=True):
            role = data.get("role", "unknown")
            label = data.get("name") or data.get("sentence") or node_id
            if role == "hyperedge":
                if len(label) > 60:
                    label = label[:57] + "..."
            elif role == "asset":
                label = data.get("path", node_id)
                label = Path(label).name if label else node_id
            nodes.append({
                "id": node_id,
                "label": str(label)[:80],
                "name_zh": data.get("name_zh", ""),
                "type": role,
                "modality": data.get("modality") or data.get("modalities", ""),
                "degree": degree_map.get(node_id, 0),
                "source": data.get("source", ""),
            })

        links = [{"source": s, "target": t} for s, t in g.edges()]
        return {"nodes": nodes, "links": links}

    def stats(self) -> Dict[str, Any]:
        from knowledgeclaw.storage import JsonKVStorage
        processed_kv = JsonKVStorage(str(Path(self._storage_dir) / "processed_files.json"))
        s = {
            "files_ingested": len(processed_kv),
            "chunks": len(self._kv),
            "entity_nodes": len(self._graph.nodes_by_role("entity")),
            "hyperedge_nodes": len(self._graph.nodes_by_role("hyperedge")),
            "asset_nodes": len(self._graph.nodes_by_role("asset")),
            "total_nodes": self._graph.number_of_nodes(),
            "total_edges": self._graph.number_of_edges(),
            "entity_vectors": len(self._vdb_entity),
            "hyperedge_vectors": len(self._vdb_hyperedge),
            "media_vectors": len(self._vdb_media),
            "community_vectors": len(self._vdb_community),
            "communities": len(self.list_communities()),
            "storage_dir": self._storage_dir,
            "kb_dir": self._kb_dir,
        }
        return s
