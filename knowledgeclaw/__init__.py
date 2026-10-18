"""KnowledgeClaw: agentic knowledge engineering over an omnimodal typed hypergraph.

  C1  perceive.py, extract.py, hypergraph.py   omnimodal typed hypergraph
  C2  lifecycle.py                             closed-loop lifecycle (Add / Del / Refresh, audit, invariant)
  C3  retrieve.py, community.py, router.py     multi-granular retrieval with intent routing
      agent.py                                 the agent facade tying the three together
"""
from knowledgeclaw.agent import KnowledgeClaw
from knowledgeclaw.backend import Backend, OpenAIBackend
from knowledgeclaw.perceive import Source

__all__ = ["KnowledgeClaw", "Backend", "OpenAIBackend", "Source"]
