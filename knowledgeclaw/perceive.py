"""C1, reading five modalities: one reader per modality turns a source file into segments.

A segment is a text surface that keeps the modality of the file it was read from (the caption of an image
is an image segment). Text and PDF are read from their text projection; image, audio and video are read
natively by the omnimodal model from the raw file, once, at ingest -- after ingest nothing reads a raw file.
"""
from __future__ import annotations

import io
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from knowledgeclaw.backend import Backend, audio_part, image_part

SEG_CHARS = 4000            # ~1000 tokens per segment
SEG_OVERLAP = 200
MAX_SIDE = 1536             # images are downscaled to this longest side before the model sees them
FRAME_EVERY_S = 20.0        # video keyframe spacing
MAX_FRAMES = 6

MEDIA = ("image", "audio", "video")


@dataclass
class Source:
    name: str                       # the file's handle (released file name)
    modality: str                   # text | pdf | image | audio | video
    text: str = ""                  # text projection, headed "[<modality> file <name>]"
    path: Path | None = None        # raw file, read natively for media
    revises: str | None = None      # handle of the file this source is a new version of (Refresh)


@dataclass
class Perceived:
    source: Source
    segments: list[str] = field(default_factory=list)
    description: str = ""           # native media description (media only)
    native: bool = False            # True when the raw media was read by the model


IMAGE_PROMPT = """You are indexing one image for a personal knowledge base, so that it can later be told apart \
from similar images and matched to its catalogue record. Write plain prose (at most 250 words), then one line \
"Visible text:" with every legible word verbatim (or "none").
Cover, when present: what kind of object the image is (photograph, photomicrograph, electron micrograph, \
engraving, drawing, painting, portrait, map, chart, floor plan, diagram, screenshot, poster, document scan, \
object photo, ...); what it depicts -- people (how many, age group, sex, activity, clothing), organisms, \
anatomy, objects, instruments, buildings, places, setting and apparent period; technique (stain, \
magnification, colour or black-and-white).
For a floor plan or technical diagram, add a line "Inventory:" that counts what is drawn, by type: rooms \
(each room type with its count), walls, windows, doors, stairs, columns, railings, and every fixed fitting or \
furniture item (sinks, toilets, showers, bathtubs, cabinets, closets, stoves, appliances, sauna benches, ...) \
with its count.
Do not guess names, places or dates that are not visible."""

AUDIO_PROMPT = """You are indexing one audio recording for a personal knowledge base. First give a 1-3 \
sentence summary: what the recording is (lecture, interview, talk, song, sound effect, ...), who speaks \
(names if said), and the topic or musical character (genre, instruments, mood, tempo). Then write "Transcript:" \
and transcribe all speech verbatim, with a time mark such as [0:30] about every 30 seconds. If there is no \
speech, write "Transcript: none"."""

VIDEO_PROMPT = """You are indexing one video for a personal knowledge base. You are given keyframes (with their \
times) and the audio track. First describe what the video shows in 2-4 sentences: kind of video (lecture, \
interview, demonstration, gameplay, animation, ...), people, setting, and all on-screen text verbatim. Then \
write "Transcript:" and transcribe all speech verbatim, with a time mark such as [0:30] about every 30 \
seconds (or "Transcript: none")."""


def split(text: str, size: int = SEG_CHARS, overlap: int = SEG_OVERLAP) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        if j < len(text):                           # end at a paragraph or sentence break when one is near
            k = max(text.rfind("\n\n", i + size // 2, j), text.rfind(". ", i + size // 2, j))
            j = k + 1 if k > 0 else j
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return out


def _image_bytes(path: Path) -> tuple[bytes, str]:
    from PIL import Image
    im = Image.open(path)
    im.load()
    if im.mode not in ("RGB", "L"):
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im.convert("RGBA"), mask=im.convert("RGBA").split()[-1])
        im = bg
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue(), "image/jpeg"


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, timeout=600)


def _wav16k(path: Path) -> bytes:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "a.wav"
        _ffmpeg(["-i", str(path), "-vn", "-ac", "1", "-ar", "16000", str(out)])
        return out.read_bytes()


def _duration(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                        str(path)], capture_output=True, text=True, timeout=60)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def _frames(path: Path) -> list[tuple[float, bytes]]:
    dur = _duration(path)
    times = [min(t, max(dur - 0.5, 0)) for t in [FRAME_EVERY_S * (i + 0.5) for i in range(MAX_FRAMES)]
             if t < dur] or [0.0]
    out = []
    with tempfile.TemporaryDirectory() as d:
        for i, t in enumerate(times):
            f = Path(d) / f"{i}.jpg"
            try:
                _ffmpeg(["-ss", f"{t:.2f}", "-i", str(path), "-frames:v", "1", "-vf", "scale=512:-2", str(f)])
                out.append((t, f.read_bytes()))
            except Exception:
                continue
    return out


def _mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def perceive(src: Source, llm: Backend) -> Perceived:
    head, _, body = src.text.partition("\n")
    head = head or f"[{src.modality} file {src.name}]"
    if src.modality not in MEDIA or src.path is None or not src.path.exists():
        return Perceived(src, [f"{head}\n{s}" for s in split(body)] or [head])
    desc = ""
    try:
        if src.modality == "image":
            data, mime = _image_bytes(src.path)
            desc = llm.chat([{"role": "user", "content": [{"type": "text", "text": IMAGE_PROMPT},
                                                          image_part(data, mime)]}],
                            json_mode=False, max_tokens=1000) or ""
        elif src.modality == "audio":
            desc = llm.chat([{"role": "user", "content": [{"type": "text", "text": AUDIO_PROMPT},
                                                          audio_part(_wav16k(src.path))]}],
                            json_mode=False, max_tokens=3000) or ""
        else:
            parts = [{"type": "text", "text": VIDEO_PROMPT}]
            for t, jpg in _frames(src.path):
                parts += [{"type": "text", "text": f"Frame at {_mmss(t)}:"}, image_part(jpg, "image/jpeg")]
            parts += [{"type": "text", "text": "Audio track:"}, audio_part(_wav16k(src.path))]
            desc = llm.chat([{"role": "user", "content": parts}], json_mode=False, max_tokens=3000) or ""
    except Exception:
        desc = ""
    desc = desc.strip()
    native = bool(desc)
    if not native:                                    # the raw file could not be read: fall back to its projection
        desc = body.strip()
    return Perceived(src, [f"{head}\n{s}" for s in split(desc)] or [head], description=desc, native=native)
