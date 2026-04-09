"""API routes for KnowledgeClaw knowledge base operations."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()


def _get_kc(request: Request):
    kc = getattr(request.app.state, "oc", None)
    if kc is None:
        raise HTTPException(status_code=503, detail="KnowledgeClaw engine not initialised")
    return kc


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    folder: str


class UpdateRequest(BaseModel):
    file: str
    force: bool = False


class DeleteRequest(BaseModel):
    source: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 10


class InferRequest(BaseModel):
    topic: str
    max_rounds: int = 5


class ResearchRequest(BaseModel):
    topic: str


class TranslateRequest(BaseModel):
    labels: List[str]
    target_lang: str = "zh"


class LifecycleRequest(BaseModel):
    actions: List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# CONSTRUCTION / UPDATE / DELETE / CONSISTENCY (C2)
# ---------------------------------------------------------------------------

@router.post("/api/ingest")
async def api_ingest(req: IngestRequest, request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content={"status": "ok", "result": kc.lifecycle.construct(req.folder)})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingest error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/update")
async def api_update(req: UpdateRequest, request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content={"status": "ok", "result": kc.lifecycle.update(req.file, force=req.force)})
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Update error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/delete")
async def api_delete(req: DeleteRequest, request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content={"status": "ok", "result": kc.lifecycle.delete(req.source)})
    except Exception as exc:
        logger.exception("Delete error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/consistency")
async def api_consistency(request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content={"status": "ok", "result": kc.consistency_check()})
    except Exception as exc:
        logger.exception("Consistency error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/communities/build")
async def api_communities_build(request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content={"status": "ok", "result": kc.lifecycle.rebuild_communities()})
    except Exception as exc:
        logger.exception("Community build error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/communities")
async def api_communities_list(request: Request):
    kc = _get_kc(request)
    return JSONResponse(content={"communities": kc.list_communities()})


# ---------------------------------------------------------------------------
# QUERY / INFER / RESEARCH / EVAL (C3)
# ---------------------------------------------------------------------------

@router.post("/api/query")
async def api_query(req: QueryRequest, request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content={"status": "ok", "result": kc.query(req.question, top_k=req.top_k)})
    except Exception as exc:
        logger.exception("Query error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/infer")
async def api_infer(req: InferRequest, request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content={"status": "ok", "result": kc.infer(req.topic, max_rounds=req.max_rounds)})
    except Exception as exc:
        logger.exception("Infer error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/evaluate")
async def api_evaluate(request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content={"status": "ok", "result": kc.evaluate()})
    except Exception as exc:
        logger.exception("Evaluate error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/research")
async def api_research(req: ResearchRequest, request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content={"status": "ok", "result": kc.research(req.topic)})
    except Exception as exc:
        logger.exception("Research error")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Lifecycle batch + audit log
# ---------------------------------------------------------------------------

@router.post("/api/lifecycle")
async def api_lifecycle(req: LifecycleRequest, request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content={"status": "ok", "result": kc.lifecycle.run_lifecycle(req.actions)})
    except Exception as exc:
        logger.exception("Lifecycle error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/lifecycle/history")
async def api_lifecycle_history(request: Request, limit: int = 50):
    kc = _get_kc(request)
    return JSONResponse(content={"history": kc.lifecycle.history(limit=limit)})


# ---------------------------------------------------------------------------
# Graph / stats / reset / translate (helpers)
# ---------------------------------------------------------------------------

@router.get("/api/graph")
async def api_graph(request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content=kc.graph_data())
    except Exception as exc:
        logger.exception("Graph data error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/api/stats")
async def api_stats(request: Request):
    kc = _get_kc(request)
    try:
        return JSONResponse(content=kc.stats())
    except Exception as exc:
        logger.exception("Stats error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/reset")
async def api_reset(request: Request):
    """Wipe all knowledge-base data and reinitialise storage."""
    import shutil
    from pathlib import Path
    kc = _get_kc(request)
    try:
        storage_dir = Path(kc._storage_dir)
        if storage_dir.exists():
            shutil.rmtree(storage_dir)
        storage_dir.mkdir(parents=True, exist_ok=True)
        kc._init_storage()
        kc._rebuild_agents()
        return JSONResponse(content={"ok": True})
    except Exception as exc:
        logger.exception("Reset error")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/api/translate")
async def api_translate(req: TranslateRequest):
    """Batch-translate a list of labels using the LLM."""
    import json
    import re
    import config as cfg
    from openai import OpenAI

    labels = list(dict.fromkeys(req.labels))
    if not labels:
        return JSONResponse(content={"translations": {}})

    BATCH = 80
    translations: Dict[str, str] = {}
    client = OpenAI(api_key=cfg.OPENAI_API_KEY, base_url=cfg.OPENAI_BASE_URL)

    for i in range(0, len(labels), BATCH):
        batch = labels[i: i + BATCH]
        prompt = (
            "Translate the following entity names to Chinese. "
            "Return ONLY a JSON object mapping each original name to its Chinese translation. "
            "Keep proper nouns or abbreviations that have no natural Chinese equivalent in their original form. "
            "Do NOT include any explanation or markdown, only the raw JSON object.\n\n"
            "Names:\n" + json.dumps(batch, ensure_ascii=False)
        )
        try:
            resp = client.chat.completions.create(
                model=cfg.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            batch_result = json.loads(raw)
            if isinstance(batch_result, dict):
                translations.update(batch_result)
        except Exception as exc:
            logger.warning("Translation batch failed: %s", exc)
            for lbl in batch:
                translations[lbl] = lbl

    return JSONResponse(content={"translations": translations})
