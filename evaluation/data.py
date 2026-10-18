"""OmniKE-Bench as the evaluation reads it: the corpus as documents, the questions, and the gold.

Expected layout under DATA_DIR (default ./data):
    corpus/<persona>/{text,pdf,image,audio,video}/<file>     base corpus; T5 increments in corpus/<persona>/deltas/
    release/CORPUS.json                                      the evaluated files of each persona
    release/questions/<task>/<persona>.jsonl                 questions (no answers)
    release/gold/<task>/<persona>.jsonl                      gold
    bridge/                                                  shared text projection of image/audio/video

Protocol: a document is one released file, shown as text headed by its own file name ("[pdf file <name>]");
text and pdf files are cut to their opening DOC_TOKENS tokens at the last paragraph break (sentence end,
failing that) before the limit; image / audio / video carry the shared text projection as their text and
are read natively from the raw file by KnowledgeClaw.
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(os.environ.get("DATA_DIR", ROOT / "data")).resolve()
CORPUS = DATASET / "corpus"
RELEASE = DATASET / "release"
BRIDGE = DATASET / "bridge"
TEXT_CACHE = ROOT / "results" / ".corpus"

PERSONAS = ["healthcare", "finance", "education", "law", "psychology",
            "politics", "humanities", "building", "music", "games"]
SHORT = {"healthcare": "HC", "finance": "Fin", "education": "Edu", "law": "Law", "psychology": "Psy",
         "politics": "Pol", "humanities": "Hum", "building": "Bld", "music": "Mus", "games": "Gam"}
TASKS = ["T1", "T2", "T3", "T4", "T5", "T6"]
MODS = ["text", "pdf", "image", "audio", "video"]

DOC_TOKENS = 4000
CHARS_PER_TOKEN = 4


@dataclass
class Doc:
    id: str            # file stem (content hash)
    file: str          # released file name, the handle items use
    modality: str
    text: str          # text projection, headed by the file name


def soft_cut(text: str, max_tokens: int = DOC_TOKENS) -> str:
    """The opening max_tokens tokens, ending at the last paragraph break (else sentence end) before the limit."""
    lim = max_tokens * CHARS_PER_TOKEN
    if len(text) <= lim:
        return text
    head = text[:lim]
    for pat in (r"\n\s*\n", r"(?<=[.!?])\s"):
        cuts = [m.end() for m in re.finditer(pat, head)]
        if cuts and cuts[-1] > lim // 2:
            return head[:cuts[-1]].rstrip()
    return head


def _pdf_text(path: Path) -> str:
    import pymupdf
    out, n = [], 0
    with pymupdf.open(str(path)) as d:
        for pg in d:
            t = pg.get_text("text")
            out.append(t)
            n += len(t)
            if n > DOC_TOKENS * CHARS_PER_TOKEN * 1.5:        # enough for soft_cut
                break
    return "\n\n".join(out)


def _bridge(persona: str, stem: str, modality: str) -> str:
    if modality == "image":
        p = BRIDGE / "gpt" / "image" / persona / f"{stem}.json"
        return json.loads(p.read_text()).get("description", "") if p.exists() else ""
    p = BRIDGE / "mimo" / modality / persona / f"{stem}.consolidated.json"
    if not p.exists():
        return ""
    return "\n".join(f"[{w['t0']:.0f}-{w['t1']:.0f}s] {w['text']}" for w in json.loads(p.read_text()).get("windows", []))


def file_text(path: Path, persona: str) -> str:
    m = path.parent.name if path.parent.name in MODS else ("pdf" if path.suffix == ".pdf" else "text")
    if m == "text" or path.suffix == ".txt":
        body = soft_cut(path.read_text(encoding="utf-8", errors="replace"))
    elif m == "pdf" or path.suffix == ".pdf":
        body = soft_cut(_pdf_text(path))
    else:
        body = _bridge(persona, path.stem, m)
    return f"[{m} file {path.name}]\n{body}"


@functools.lru_cache(maxsize=16)
def corpus(persona: str) -> tuple[Doc, ...]:
    """The persona's evaluated base corpus (release/CORPUS.json). The text projection is cached on disk."""
    raw = (RELEASE / "CORPUS.json").read_bytes()
    tag = hashlib.sha256(raw + f"{DOC_TOKENS}".encode()).hexdigest()[:12]
    cf = TEXT_CACHE / f"{persona}.{tag}.json"
    if cf.exists():
        return tuple(Doc(**d) for d in json.loads(cf.read_text()))
    spec = json.loads(raw)["personas"][persona]["files"]
    docs = []
    for m in MODS:
        for name in spec[m]:
            p = CORPUS / persona / m / name
            docs.append(Doc(p.stem, name, m, file_text(p, persona)))
    TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cf.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps([d.__dict__ for d in docs], ensure_ascii=False))
    tmp.replace(cf)
    return tuple(docs)


def media_path(persona: str, modality: str, file_name: str) -> Path:
    return CORPUS / persona / modality / file_name


def delta_docs(q: dict) -> list[Doc]:
    """A T5 item's increment: the next version of its base document, released apart from the corpus."""
    out, obj = [], q["question"].get("object")
    for rel in q["question"].get("delta_paths") or []:
        p = DATASET / rel
        if p.exists():
            head, _, body = file_text(p, q["persona"]).partition("\n")
            # the increment says which released object it revises (the item names the object, not the delta)
            head = f"{head[:-1]}, a new version of {obj}]" if obj else head
            out.append(Doc(p.stem, p.name, "pdf" if p.suffix == ".pdf" else "text", f"{head}\n{body}"))
    return out


def _jsonl(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def items(task: str, persona: str) -> list[tuple[dict, dict]]:
    """(question, gold) pairs. The method is handed the question only; the gold is kept for the scorer."""
    gold = {g["qid"]: g for g in _jsonl(RELEASE / "gold" / task / f"{persona}.jsonl")}
    return [(q, gold[q["qid"]]) for q in _jsonl(RELEASE / "questions" / task / f"{persona}.jsonl")]
