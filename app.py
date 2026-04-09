"""
KnowledgeClaw FastAPI application.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

from routers import kb as kb_router

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the KnowledgeClaw engine on startup."""
    import config as cfg
    from knowledgeclaw import KnowledgeClaw

    kb_dir = os.getenv("KNOWLEDGECLAW_KB_DIR", str(cfg.BASE_DIR / "kb"))
    storage_dir = os.getenv("KNOWLEDGECLAW_STORAGE_DIR", str(cfg.BASE_DIR / "kb" / ".knowledgeclaw"))

    Path(kb_dir).mkdir(parents=True, exist_ok=True)
    Path(storage_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Initialising KnowledgeClaw – kb_dir=%s, storage_dir=%s", kb_dir, storage_dir)
    app.state.oc = KnowledgeClaw(kb_dir=kb_dir, storage_dir=storage_dir)
    logger.info("KnowledgeClaw ready")

    yield

    # Cleanup on shutdown
    logger.info("KnowledgeClaw shutting down")


app = FastAPI(title="KnowledgeClaw", version="0.1.0", lifespan=lifespan)

# Static files
_static_dir = Path(__file__).parent / "static"
_templates_dir = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
templates = Jinja2Templates(directory=str(_templates_dir))

# Routers
app.include_router(kb_router.router)


# ---------------------------------------------------------------------------
# Page routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.get("/graph", response_class=HTMLResponse)
async def graph_page(request: Request):
    """Graph visualisation page."""
    return templates.TemplateResponse("graph.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Chat / Q&A interface."""
    return templates.TemplateResponse("chat.html", {"request": request})
