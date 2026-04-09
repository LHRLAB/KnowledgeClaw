"""Evaluation config — the LLM and embedding model used at eval time.

All systems are evaluated with the same backbone to ensure fairness.
"""

EVAL_LLM = "gemini-3.1-flash-lite-preview"
EVAL_EMBED = "gemini-embedding-2-preview"
EVAL_EMBED_DIM = 3072
