"""subalign 功能测试台 — FastAPI backend.

Every job runs the real CLI (``subalign.cli.main``) in-process with the flags
built by the page, so the page exercises exactly what the command line does.

    python -m webui.server            # http://127.0.0.1:7860
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import mimetypes
import logging
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths  # noqa: F401  (must run before model libraries are imported)
from . import ollama

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT = paths.ROOT
JOBS_DIR = paths.WORK / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).resolve().parent / "static"

log = logging.getLogger("webui")
app = FastAPI(title="subalign 功能测试台")
mimetypes.add_type("application/wasm", ".wasm")   # Windows registries often lack it
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# CJK fallback font for the libass (ASS effect) preview: a system font, served, not copied
_CJK_FONTS = [Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / n
              for n in ("msyh.ttc", "simhei.ttf", "Deng.ttf", "simsun.ttc")]
_CJK_FONTS += [Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"), Path("/System/Library/Fonts/PingFang.ttc")]

# ------------------------------------------------------------------ model caches
# The CLI builds a new ASR / CTC model per call; keep them loaded between jobs.
_model_cache: Dict[Any, Any] = {}
_cache_lock = threading.Lock()


def _install_model_cache() -> None:
    import subalign.asr as asr_mod
    import subalign.align.ctc as ctc_mod

    orig_get_backend = asr_mod.get_backend
    orig_emitter = ctc_mod.HFCTCEmitter

    def cached_get_backend(name: str = "auto", **kw):
        key = ("asr", name, tuple(sorted(kw.items())))
        with _cache_lock:
            if key not in _model_cache:
                _model_cache[key] = orig_get_backend(name, **kw)
            return _model_cache[key]

    def cached_emitter(model=None, language=None, device=None):
        key = ("ctc", model, (language or "").split("-")[0], device)
        with _cache_lock:
            if key not in _model_cache:
                _model_cache[key] = orig_emitter(model, language, device)
            return _model_cache[key]

    asr_mod.get_backend = cached_get_backend
    ctc_mod.HFCTCEmitter = cached_emitter


def free_models() -> int:
    with _cache_lock:
        n = len(_model_cache)
        _model_cache.clear()
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return n


# ------------------------------------------------------------------ jobs
class Job:
    def __init__(self, task: str, argv: List[str], workdir: Path, meta: Dict[str, Any]):
        self.id = workdir.name
        self.task = task
        self.argv = argv
        self.workdir = workdir
        self.meta = meta
        self.status = "queued"
        self.log: List[str] = []
        self.stdout = ""
        self.error: Optional[str] = None
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None

    def to_dict(self, full: bool = True) -> Dict[str, Any]:
        d = {"id": self.id, "task": self.task, "status": self.status, "created": self.created,
             "started": self.started, "finished": self.finished, "error": self.error,
             "command": ("subalign " + " ".join(shlex.quote(_rel(a)) for a in _masked(self.argv))
                         if self.argv else self.meta.get("_command", "")),
             "meta": self.meta}
        if full:
            d["log"] = "\n".join(self.log[-2000:])
            d["stdout"] = self.stdout
            d["files"] = self.files() if self.status in ("done", "error") else []
        return d

    def files(self) -> List[Dict[str, Any]]:
        out = []
        base = self.workdir / "output"
        if not base.exists():
            return out
        for p in sorted(base.rglob("*")):
            if p.is_file() and "work" not in p.relative_to(base).parts[:-1] and not p.name.startswith("."):
                rel = p.relative_to(self.workdir).as_posix()
                out.append({"name": p.relative_to(base).as_posix(), "url": f"/jobs/{self.id}/file/{rel}",
                            "size": p.stat().st_size, "kind": _file_kind(p)})
        # separated stems written by the lyric pipeline live in output/work
        work = base / "work"
        if work.exists():
            for p in sorted(work.glob("*.*")):
                if p.is_file() and p.suffix.lower() in (".wav", ".flac", ".mp3"):
                    rel = p.relative_to(self.workdir).as_posix()
                    out.append({"name": "work/" + p.name, "url": f"/jobs/{self.id}/file/{rel}",
                                "size": p.stat().st_size, "kind": "audio"})
        return out


def _masked(argv: List[str]) -> List[str]:
    out = list(argv)
    for i, a in enumerate(out[:-1]):
        if a == "--llm-api-key" and out[i + 1]:
            out[i + 1] = "***"
    return out


def _rel(a: str) -> str:
    try:
        return str(Path(a).relative_to(ROOT)) if os.path.isabs(a) else a
    except ValueError:
        return a


def _file_kind(p: Path) -> str:
    ext = p.suffix.lower()
    if ext in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        return "audio"
    if ext in (".mp4", ".mkv", ".mov", ".webm", ".m4v"):
        return "video"
    if ext == ".krc" and not p.name.endswith(".krc.txt"):
        return "binary"
    if ext == ".json":
        return "json"
    if ext == ".md":
        return "markdown"
    return "text"


JOBS: Dict[str, Job] = {}
_queue: "queue.Queue[Job]" = queue.Queue()


class _JobLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.job: Optional[Job] = None
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record):
        if self.job is not None:
            try:
                self.job.log.append(self.format(record))
            except Exception:
                pass


_handler = _JobLogHandler()


def _worker() -> None:
    from subalign.cli import main as cli_main

    logging.getLogger().addHandler(_handler)
    for name in ("subalign", "webui", "faster_whisper", "audio_separator"):
        logging.getLogger(name).setLevel(logging.INFO)
    while True:
        job = _queue.get()
        job.status = "running"
        job.started = time.time()
        _handler.job = job
        buf = io.StringIO()
        try:
            job.log.append("$ " + job.to_dict(False)["command"])
            with contextlib.redirect_stdout(buf):
                code = cli_main(job.argv)
            job.stdout = buf.getvalue()
            job.status = "done" if code == 0 else "error"
            if code:
                job.error = f"exit code {code}"
        except SystemExit as e:  # argparse errors
            job.stdout = buf.getvalue()
            job.status = "error"
            job.error = f"参数错误 (exit {e.code})"
        except BaseException as e:
            job.stdout = buf.getvalue()
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            job.log.append(traceback.format_exc())
        finally:
            job.finished = time.time()
            job.log.append(f"-- {job.status} in {job.finished - job.started:.1f}s")
            _handler.job = None
            (job.workdir / "job.json").write_text(json.dumps({**job.to_dict(), "argv": job.argv},
                                                             ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ helpers
TASKS = ("align", "lyrics", "separate", "roughcut", "convert", "translate", "proofread")


def _allowed_flags(task: str) -> set:
    from subalign.cli import build_parser

    p = build_parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
    sp = sub.choices[task]
    return {s for a in sp._actions for s in a.option_strings}


async def _save_upload(f: UploadFile, dst_dir: Path) -> Path:
    name = Path(f.filename or "upload.bin").name
    dst = dst_dir / name
    with open(dst, "wb") as out:
        while True:
            chunk = await f.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return dst


# ------------------------------------------------------------------ routes
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.post("/api/jobs")
async def create_job(task: str = Form(...), args: str = Form("[]"), media: Optional[UploadFile] = File(None),
                     text_file: Optional[UploadFile] = File(None), text_content: str = Form(""),
                     text_name: str = Form(""), glossary: str = Form(""), style_json: str = Form(""),
                     sample: str = Form("")):
    if task not in TASKS:
        raise HTTPException(400, f"unknown task {task}")
    wd = _new_workdir()
    meta: Dict[str, Any] = {}
    argv: List[str] = [task]

    # --- primary input
    media_path: Optional[Path] = None
    if task in ("align", "lyrics", "separate", "roughcut"):
        if media is not None and media.filename:
            media_path = await _save_upload(media, wd / "input")
        elif sample:
            src = paths.WORK / "samples" / Path(sample).name
            if not src.exists():
                raise HTTPException(400, "sample not found")
            media_path = wd / "input" / src.name
            shutil.copy(src, media_path)
        else:
            raise HTTPException(400, "请上传音频/视频文件")
        argv.append(str(media_path))
        meta["media"] = f"/jobs/{wd.name}/file/input/{media_path.name}"

    # --- secondary text input (script / lyrics / subtitle)
    text_path: Optional[Path] = None
    if text_file is not None and text_file.filename:
        text_path = await _save_upload(text_file, wd / "input")
    elif text_content.strip():
        name = Path(text_name or ("script.txt" if task in ("align", "lyrics") else "input.srt")).name
        text_path = wd / "input" / name
        text_path.write_text(text_content, encoding="utf-8")
    if task in ("align", "roughcut") and text_path:
        argv += ["--script", str(text_path)]
    elif task == "lyrics" and text_path:
        argv += ["--lyrics", str(text_path)]
    elif task in ("convert", "translate", "proofread"):
        if not text_path:
            raise HTTPException(400, "请上传字幕文件或粘贴字幕内容")
        argv.append(str(text_path))
    if text_path:
        meta["text_input"] = text_path.name

    try:
        pairs = json.loads(args)
    except json.JSONDecodeError:
        raise HTTPException(400, "args must be JSON")
    argv = _add_flags(task, argv, pairs, wd, glossary, style_json)
    # remembered so that a corrected transcript can be re-aligned with the same settings
    meta.update(pairs=pairs, glossary=glossary, style_json=style_json)
    return _enqueue(task, argv, wd, meta)


def _new_workdir() -> Path:
    wd = JOBS_DIR / (time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6])
    (wd / "input").mkdir(parents=True)
    return wd


def _add_flags(task: str, argv: List[str], pairs: List[list], wd: Path, glossary: str, style_json: str) -> List[str]:
    """Append glossary / form flags / style / output dir to ``argv``."""
    argv = list(argv)
    if glossary.strip():
        try:
            json.loads(glossary)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"术语表 JSON 无效: {e}")
        gp = wd / "input" / "glossary.json"
        gp.write_text(glossary, encoding="utf-8")
        argv += ["--glossary", str(gp)]
    allowed = _allowed_flags(task)
    has_style = False
    for item in pairs:
        flag, val = item[0], item[1] if len(item) > 1 else None
        if flag not in allowed or flag in ("-o", "--out", "--script", "--lyrics", "--glossary"):
            raise HTTPException(400, f"flag {flag} not allowed for {task}")
        if flag == "--style":
            has_style = True
        if val is True or val is None:
            argv.append(flag)
        elif val is False or val == "":
            continue
        else:
            argv += [flag, str(val)]
    if style_json.strip() and "--style" in allowed:
        try:
            json.loads(style_json)
        except json.JSONDecodeError as e:
            raise HTTPException(400, f"样式 JSON 无效: {e}")
        sp = wd / "input" / "style.json"
        sp.write_text(style_json, encoding="utf-8")
        if has_style:
            i = argv.index("--style")
            argv[i + 1] = str(sp)
        else:
            argv += ["--style", str(sp)]
    # the page previews the json output of alignments
    if task in ("align", "lyrics", "convert", "roughcut") and "-f" in argv:
        i = argv.index("-f")
        fl = argv[i + 1].split(",")
        if "json" not in fl:
            argv[i + 1] = ",".join(fl + ["json"])
    return argv + ["-o", str(wd / "output")]


def _enqueue(task: str, argv: List[str], wd: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    job = Job(task, argv, wd, meta)
    JOBS[job.id] = job
    _queue.put(job)
    return {"id": job.id, "command": job.to_dict(False)["command"]}


def _load_job(job_id: str) -> Optional[Job]:
    """A job from this run, or one restored from its job.json (after a restart)."""
    if job_id in JOBS:
        return JOBS[job_id]
    f = JOBS_DIR / Path(job_id).name / "job.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text(encoding="utf-8"))
    job = Job(d["task"], d.get("argv") or [], f.parent, dict(d.get("meta") or {}, _command=d.get("command", "")))
    job.status, job.error = d.get("status", "done"), d.get("error")
    job.created, job.started, job.finished = d.get("created", 0), d.get("started"), d.get("finished")
    job.log, job.stdout = (d.get("log") or "").splitlines(), d.get("stdout", "")
    JOBS[job.id] = job
    return job


MEDIA_EXT = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma",
             ".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi", ".flv", ".ts"}


def _fmt_lrc_time(t: float) -> str:
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:05.2f}"


@app.post("/api/jobs/{job_id}/realign")
def realign(job_id: str, body: Dict[str, Any] = Body(...)):
    """Re-run an alignment with corrected text.

    ``body.lines``: [{"start": seconds|null, "text": str, "translation": str|null}].
    The corrected lines become a timed LRC script: the old line times then act as a
    cross-check / fallback for CTC, and the separated vocals of the parent job are
    reused so the song is not separated again.
    """
    parent = _load_job(job_id)
    if parent is None or parent.task not in ("align", "lyrics"):
        raise HTTPException(404, "job not found")
    lines = [ln for ln in body.get("lines", []) if str(ln.get("text", "")).strip()]
    if not lines:
        raise HTTPException(400, "没有歌词 / 文本")
    meta_p = parent.meta or {}
    wd = _new_workdir()
    rows = []
    # keep title / artist so that the "Artist - Title" header is still recognised as a credit line
    tags = {"title": "ti", "artist": "ar", "album": "al"}
    for k, v in (body.get("metadata") or {}).items():
        if k in tags and str(v).strip():
            rows.append(f"[{tags[k]}:{' '.join(str(v).split())}]")
    for ln in lines:
        text = " ".join(str(ln["text"]).split())
        st = ln.get("start")
        rows.append(f"[{_fmt_lrc_time(max(0.0, float(st)))}]{text}" if st is not None else text)
    script = wd / "input" / "edited.lrc"
    script.write_text("\n".join(rows) + "\n", encoding="utf-8")

    src_media = Path(parent.argv[1]) if len(parent.argv) > 1 else None
    if src_media is None or not src_media.exists():   # jobs saved before argv was recorded
        src_media = next((f for f in sorted((parent.workdir / "input").glob("*"))
                          if f.suffix.lower() in MEDIA_EXT), None)
    if src_media is None or not src_media.exists():
        raise HTTPException(400, "原任务的音频已不存在")
    pairs = [p for p in meta_p.get("pairs", []) if p and p[0] != "--separation"]
    work = parent.workdir / "output" / "work"
    vocals = next(iter(sorted(work.glob("*.vocals.wav"))), None) if work.exists() else None
    if vocals is not None:
        # analyse the already separated vocals; keep the original base name for the outputs
        media = wd / "input" / (src_media.stem + ".wav")
        shutil.copy(vocals, media)
        pairs.append(["--separation", "none"])
        if parent.task == "align" and not any(p[0] == "--mode" for p in pairs):
            pairs.append(["--mode", "song"])
    else:
        media = wd / "input" / src_media.name
        shutil.copy(src_media, media)
    argv = [parent.task, str(media), "--script" if parent.task == "align" else "--lyrics", str(script)]
    argv = _add_flags(parent.task, argv, pairs, wd, meta_p.get("glossary", ""), meta_p.get("style_json", ""))
    meta = dict({k: v for k, v in meta_p.items() if k != "_command"}, pairs=pairs, parent=parent.id, edited=True,
                text_input=script.name,
                media=meta_p.get("media") or f"/jobs/{wd.name}/file/input/{media.name}")
    return _enqueue(parent.task, argv, wd, meta)


@app.get("/api/jobs")
def list_jobs(limit: int = 200):
    # jobs of earlier runs live on disk only: load them too
    for d in JOBS_DIR.iterdir():
        if d.name not in JOBS and (d / "job.json").exists():
            try:
                _load_job(d.name)
            except Exception:
                pass
    return [j.to_dict(False) for j in sorted(JOBS.values(), key=lambda j: -(j.created or 0))[:limit]]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _load_job(job_id)
    if not job:
        raise HTTPException(404)
    d = job.to_dict()
    d["queue_position"] = list(_queue.queue).index(job) + 1 if job.status == "queued" else 0
    return d


@app.get("/jobs/{job_id}/file/{rel:path}")
def job_file(job_id: str, rel: str):
    base = (JOBS_DIR / job_id).resolve()
    p = (base / rel).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, filename=p.name if p.suffix.lower() == ".krc" else None)


# system fonts offered in the style editor: (ASS font names, file).  libass matches the
# Fontname against the names inside the file, so both the Chinese and English names work
SYSTEM_FONTS = [
    (["微软雅黑", "Microsoft YaHei"], "msyh.ttc"),
    (["黑体", "SimHei"], "simhei.ttf"),
    (["等线", "DengXian"], "Deng.ttf"),
    (["宋体", "SimSun"], "simsun.ttc"),
    (["楷体", "KaiTi"], "simkai.ttf"),
    (["仿宋", "FangSong"], "simfang.ttf"),
    (["华文细黑", "STXihei"], "STXIHEI.TTF"),
    (["华文楷体", "STKaiti"], "STKAITI.TTF"),
    (["华文中宋", "STZhongsong"], "STZHONGS.TTF"),
    (["微軟正黑體", "Microsoft JhengHei"], "msjh.ttc"),
]
_FONT_DIR = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"


@app.get("/api/fonts")
def fonts():
    out = []
    for names, fn in SYSTEM_FONTS:
        if (_FONT_DIR / fn).exists():
            out.append({"names": names, "url": f"/fonts/sys/{fn}"})
    return out


@app.get("/fonts/sys/{fn}")
def system_font(fn: str):
    if fn not in {f for _, f in SYSTEM_FONTS} or not (_FONT_DIR / fn).exists():
        raise HTTPException(404)
    return FileResponse(_FONT_DIR / fn, media_type="font/collection" if fn.lower().endswith(".ttc") else "font/ttf")


@app.get("/fonts/cjk")
def cjk_font():
    f = next((f for f in _CJK_FONTS if f.exists()), None)
    if f is None:
        raise HTTPException(404, "no CJK font found")
    return FileResponse(f, media_type="font/collection" if f.suffix.lower() == ".ttc" else "font/ttf")


def _pair(pairs: List[list], flag: str, default=None):
    for p in pairs or []:
        if p and p[0] == flag:
            return p[1] if len(p) > 1 else True
    return default


def _style_targets(job: Job) -> List[Dict[str, Any]]:
    """Every .ass output that has a sibling .json (the document it was written from)."""
    from subalign.segment.layout import LAYOUTS

    out_dir = job.workdir / "output"
    pairs = (job.meta or {}).get("pairs", [])
    default_layout = (str(_pair(pairs, "--layout", "landscape")).split(",") or ["landscape"])[0].strip() or "landscape"
    res = []
    for ass in sorted(out_dir.glob("*.ass")):
        js = ass.with_suffix(".json")
        if not js.exists():
            continue
        last = ass.stem.rsplit(".", 1)[-1] if "." in ass.stem else ""
        is_layout = last in LAYOUTS or (("x" in last) and last.replace("x", "").isdigit())
        res.append({"ass": ass.name, "json": js.name, "layout": last if is_layout else default_layout})
    return res


def _job_style_spec(job: Job) -> Dict[str, Any]:
    """The style the job was exported with, as an editable spec."""
    from dataclasses import asdict

    from subalign.style import PRESETS, make_style

    meta = job.meta or {}
    pairs = meta.get("pairs", [])
    preset, overrides = None, {}
    if meta.get("style_edit"):
        return meta["style_edit"]
    if (meta.get("style_json") or "").strip():
        data = json.loads(meta["style_json"])
        preset, overrides = data.pop("preset", "default"), data
    else:
        st = _pair(pairs, "--style")
        if st in PRESETS:
            preset = st
    if preset is None:
        # same default as the exporter: karaoke for songs, default for speech
        js = next(iter(_style_targets(job)), None)
        kind = "speech"
        if js:
            kind = json.loads((job.workdir / "output" / js["json"]).read_text(encoding="utf-8")).get("kind", "speech")
        preset = "karaoke" if kind == "song" else "default"
    cfg = asdict(make_style(preset, overrides))
    fs = _pair(pairs, "--font-size")
    return {"preset": preset, "main": cfg["main"], "translation": cfg["translation"], "effect": cfg["effect"],
            "translation_position": cfg["translation_position"], "font_size": int(fs) if fs else None}


def _style_from_spec(spec: Dict[str, Any]):
    from subalign.style import AssStyle, Effect, make_style

    keys = lambda cls: {f for f in cls.__dataclass_fields__}  # noqa: E731
    over = {
        "main": {k: v for k, v in (spec.get("main") or {}).items() if k in keys(AssStyle) and k != "name"},
        "translation": {k: v for k, v in (spec.get("translation") or {}).items() if k in keys(AssStyle) and k != "name"},
        "effect": {k: v for k, v in (spec.get("effect") or {}).items() if k in keys(Effect)},
    }
    if spec.get("translation_position") in ("below", "above"):
        over["translation_position"] = spec["translation_position"]
    try:
        return make_style(spec.get("preset") or "default", over)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _render_ass(job: Job, target: Dict[str, Any], spec: Dict[str, Any]) -> str:
    from subalign.formats.ass import write_ass
    from subalign.models import Document
    from subalign.segment.layout import get_layout
    from subalign.segment.linebreak import segment_document

    pairs = (job.meta or {}).get("pairs", [])
    doc = Document.from_dict(json.loads((job.workdir / "output" / target["json"]).read_text(encoding="utf-8")))
    fs = spec.get("font_size") or _pair(pairs, "--font-size")
    mu, ml = _pair(pairs, "--max-chars"), _pair(pairs, "--max-lines")
    layout = get_layout(target["layout"], int(fs) if fs else None, int(mu) if mu else None, int(ml) if ml else None)
    punct = _pair(pairs, "--punct", "keep")
    if spec.get("font_size"):
        # a bigger font may no longer fit the row width: break the lines again for the new size
        doc = segment_document(doc, layout, punct=punct)
    return write_ass(doc, style=_style_from_spec(spec), layout=layout, bilingual=not _pair(pairs, "--mono", False),
                     punct=punct)


@app.get("/api/jobs/{job_id}/style")
def job_style(job_id: str):
    from subalign.style import PRESETS

    job = _load_job(job_id)
    if job is None:
        raise HTTPException(404)
    return {"spec": _job_style_spec(job), "targets": _style_targets(job), "presets": list(PRESETS)}


@app.get("/api/styles/{preset}")
def preset_style(preset: str):
    from dataclasses import asdict

    from subalign.style import PRESETS, make_style

    if preset not in PRESETS:
        raise HTTPException(404)
    cfg = asdict(make_style(preset))
    return {"preset": preset, "main": cfg["main"], "translation": cfg["translation"], "effect": cfg["effect"],
            "translation_position": cfg["translation_position"]}


@app.post("/api/jobs/{job_id}/style")
def apply_style(job_id: str, body: Dict[str, Any] = Body(...)):
    """``{spec, target, save}``: render ``target`` (an .ass name) with ``spec`` and return
    its text; with ``save`` every .ass of the job is rewritten with the new style."""
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(404)
    if job.status not in ("done", "error"):
        raise HTTPException(409, "任务还在运行")
    spec = body.get("spec") or {}
    targets = _style_targets(job)
    if not targets:
        raise HTTPException(400, "这个任务没有可重新生成的 .ass（缺少对应的 .json）")
    tgt = next((t for t in targets if t["ass"] == body.get("target")), targets[0])
    text = _render_ass(job, tgt, spec)
    if not body.get("save"):
        return {"ass": text, "target": tgt["ass"]}
    written = []
    for t in targets:
        (job.workdir / "output" / t["ass"]).write_text(text if t is tgt else _render_ass(job, t, spec), encoding="utf-8")
        written.append(t["ass"])
    # keep the style for later re-alignments of this job (and show it on reload)
    style_file = {"preset": spec.get("preset") or "default",
                  **{k: spec[k] for k in ("main", "translation", "effect", "translation_position") if k in spec}}
    for st in (style_file.get("main"), style_file.get("translation")):
        if isinstance(st, dict):
            st.pop("name", None)
    job.meta["style_json"] = json.dumps(style_file, ensure_ascii=False)
    job.meta["style_edit"] = spec
    if spec.get("font_size"):
        job.meta["pairs"] = [p for p in job.meta.get("pairs", []) if p[0] != "--font-size"] + \
            [["--font-size", int(spec["font_size"])]]
    (job.workdir / "job.json").write_text(json.dumps({**job.to_dict(), "argv": job.argv}, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    return {"ass": text, "target": tgt["ass"], "written": written}


@app.post("/api/jobs/{job_id}/roughcut")
def roughcut_apply(job_id: str, body: Dict[str, Any] = Body(...)):
    """Render a rough cut again with reviewed decisions.

    ``body.cuts``: the decision for every plan item (reason string or null), in plan order;
    ``body.settings``: optional pause settings.  Detection / recognition is not repeated.
    """
    parent = _load_job(job_id)
    if parent is None or parent.task != "roughcut":
        raise HTTPException(404, "job not found")
    plan_file = next(iter(sorted((parent.workdir / "output").glob("*.roughcut.json"))), None)
    if plan_file is None:
        raise HTTPException(400, "原任务没有剪辑计划")
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    cuts = body.get("cuts")
    if not isinstance(cuts, list) or len(cuts) != len(plan["items"]):
        raise HTTPException(400, "cuts 与剪辑计划不匹配")
    for it, c in zip(plan["items"], cuts):
        it["cut"] = c if c else None
    st = body.get("settings") or {}
    plan.setdefault("settings", {})
    for k in ("pauses", "max_pause", "keep_pause", "min_gap", "crossfade_ms"):
        if k in st and st[k] is not None:
            plan["settings"][k] = st[k]
    src_media = Path(parent.argv[1]) if len(parent.argv) > 1 else None
    if src_media is None or not src_media.exists():
        src_media = next((f for f in sorted((parent.workdir / "input").glob("*")) if f.suffix.lower() in MEDIA_EXT), None)
    if src_media is None:
        raise HTTPException(400, "原任务的音频已不存在")
    wd = _new_workdir()
    media = wd / "input" / src_media.name
    shutil.copy(src_media, media)
    pf = wd / "input" / "edited.roughcut.json"
    pf.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    meta_p = parent.meta or {}
    # settings now come from the plan; recognition / detection flags are irrelevant
    drop = {"--no-pauses", "--max-pause", "--keep-pause", "--min-gap", "--crossfade-ms", "--llm"}
    pairs = [p for p in meta_p.get("pairs", []) if p and p[0] not in drop]
    if st.get("pauses") is False:
        pairs.append(["--no-pauses", True])
    argv = _add_flags("roughcut", ["roughcut", str(media), "--plan", str(pf)], pairs, wd,
                      "", meta_p.get("style_json", ""))
    meta = {k: v for k, v in meta_p.items() if k not in ("_command", "style_edit")}
    meta.update(pairs=pairs, parent=parent.id, edited=True, media=meta_p.get("media"))
    return _enqueue("roughcut", argv, wd, meta)


@app.post("/api/models/free")
def api_free():
    return {"freed": free_models()}


# ------------------------------------------------------------------ info / status
def _have(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except Exception:
        return False


def _dir_size(p: Path) -> int:
    # hf snapshots/ are symlinks into blobs/: count each file once
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink()) if p.exists() else 0


def _local_models() -> List[Dict[str, Any]]:
    hub = paths.DIRS["HF_HOME"] / "hub"
    out = []
    if hub.exists():
        for d in sorted(hub.glob("models--*")):
            out.append({"group": "huggingface", "name": d.name[8:].replace("--", "/"), "size": _dir_size(d)})
    for f in sorted((paths.DIRS["XDG_CACHE_HOME"] / "whisper").glob("*.pt")):
        out.append({"group": "openai-whisper", "name": f.stem, "size": f.stat().st_size})
    for f in sorted((paths.DIRS["TORCH_HOME"] / "hub" / "checkpoints").glob("*")):
        out.append({"group": "demucs", "name": f.name, "size": f.stat().st_size})
    for f in sorted(paths.DIRS["SUBALIGN_UVR_MODEL_DIR"].glob("*")):
        if f.suffix in (".ckpt", ".onnx", ".pth"):
            out.append({"group": "audio-separator", "name": f.name, "size": f.stat().st_size})
    ms = paths.DIRS["MODELSCOPE_CACHE"]
    for d in sorted(ms.rglob("configuration.json")):
        out.append({"group": "modelscope", "name": d.parent.relative_to(ms).as_posix(), "size": _dir_size(d.parent)})
    om = paths.DIRS["OLLAMA_MODELS"]
    if (om / "manifests").exists():
        for m in ollama.models() or []:
            out.append({"group": "ollama", "name": m, "size": None})
        out.append({"group": "ollama", "name": "(blobs total)", "size": _dir_size(om / "blobs")})
    return out


@app.get("/api/info")
def info():
    from subalign import __version__
    from subalign.formats import FORMATS
    from subalign.llm import ANTHROPIC_DEFAULT_MODEL, OPENAI_COMPATIBLE
    from subalign.segment.layout import LAYOUTS
    from subalign.style import PRESETS
    from subalign.translate import LANG_NAMES

    cuda = None
    try:
        import torch

        cuda = {"available": torch.cuda.is_available(), "torch": torch.__version__,
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            cuda.update(free_mb=free // 2**20, total_mb=total // 2**20)
    except Exception as e:
        cuda = {"available": False, "error": str(e)}
    deps = {
        "faster-whisper (ASR)": _have("faster_whisper"),
        "openai-whisper (ASR)": _have("whisper"),
        "funasr Paraformer (ASR)": _have("funasr"),
        "torch + transformers (CTC)": _have("torch") and _have("transformers"),
        "demucs (分离)": _have("demucs"),
        "audio-separator RoFormer (分离)": _have("audio_separator"),
        "pypinyin (同音对齐)": _have("pypinyin"),
        "jieba (分词)": _have("jieba"),
        "anthropic SDK (Claude)": _have("anthropic"),
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }
    return {
        "version": __version__,
        "formats": [{"name": f.name, "ext": f.ext, "level": f.level, "read": f.reader is not None,
                     "description": f.description} for f in FORMATS.values()],
        "presets": list(PRESETS),
        "layouts": {k: {"res": f"{v.play_res_x}x{v.play_res_y}", "font_size": v.font_size,
                        "max_units": v.max_units, "max_lines": v.max_lines} for k, v in LAYOUTS.items()},
        "languages": LANG_NAMES,
        "llm_providers": {"anthropic": {"model": ANTHROPIC_DEFAULT_MODEL, "env": "ANTHROPIC_API_KEY",
                                        "has_key": bool(os.environ.get("ANTHROPIC_API_KEY"))},
                          **{k: {"model": v[1], "base_url": v[0], "env": v[2],
                                 "has_key": bool(os.environ.get(v[2]))} for k, v in OPENAI_COMPATIBLE.items()}},
        "deps": deps,
        "cuda": cuda,
        "ollama": {"installed": paths.OLLAMA_EXE.exists(), "running": ollama.running(), "models": ollama.models()},
        "models": _local_models(),
        "models_dir": str(paths.MODELS),
        "loaded_models": [list(map(str, k[:2])) for k in _model_cache],
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }


@app.post("/api/llm/test")
def llm_test(provider: str = Form("ollama"), model: str = Form(""), base_url: str = Form(""),
             api_key: str = Form("")):
    from subalign.llm import make_client

    if provider == "ollama":
        ollama.ensure_server()
    t = time.time()
    try:
        c = make_client(provider, model or None, base_url or None, api_key or None)
        reply = c.complete("You are a helpful assistant. Reply in one short sentence.",
                           "用一句话介绍你自己，并说明你是什么模型。")
        return {"ok": True, "reply": reply, "model": getattr(c, "model", ""), "seconds": round(time.time() - t, 1)}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "seconds": round(time.time() - t, 1)}


@app.post("/api/ollama/start")
def ollama_start():
    return {"running": ollama.ensure_server(), "models": ollama.models()}


# ------------------------------------------------------------------ samples
SAMPLE_LYRICS = ["我们一起走过的路", "风吹过了山岗", "你说那是最美的时光", "不要忘记"]


@app.post("/api/samples/{kind}")
def make_sample(kind: str):
    """Synthetic test audio from tests/synth.py (song: stereo with accompaniment; speech)."""
    import numpy as np
    import soundfile as sf

    sys.path.insert(0, str(ROOT / "tests"))
    from synth import SR, synth_song

    d = paths.WORK / "samples"
    d.mkdir(parents=True, exist_ok=True)
    if kind == "song":
        y, truth = synth_song([len(l) for l in SAMPLE_LYRICS], seed=3)
        t = np.arange(len(y)) / SR
        acc = 0.05 * np.sin(2 * np.pi * 330 * t)
        audio = np.vstack([y + acc, y - acc]).T
    elif kind == "speech":
        y, truth = synth_song([len(l) for l in SAMPLE_LYRICS], seed=5, speech=True, gap=0.5)
        audio = y
    else:
        raise HTTPException(404)
    p = d / f"synth_{kind}.wav"
    sf.write(p, audio, SR)
    truth_lines = [[{"text": ch, "start": round(s, 3), "end": round(e, 3)} for ch, (s, e) in zip(line, tl)]
                   for line, tl in zip(SAMPLE_LYRICS, truth)]
    return {"name": p.name, "url": f"/samples/{p.name}", "text": "\n".join(SAMPLE_LYRICS), "truth": truth_lines}


@app.get("/samples/{name}")
def sample_file(name: str):
    p = paths.WORK / "samples" / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


# ------------------------------------------------------------------ unit tests
_pytest_state: Dict[str, Any] = {"running": False, "output": "", "code": None}


@app.post("/api/pytest")
def run_pytest():
    if _pytest_state["running"]:
        return _pytest_state

    def go():
        _pytest_state.update(running=True, output="", code=None)
        p = subprocess.Popen([sys.executable, "-m", "pytest", "-v", "--color=no", "-p", "no:cacheprovider"],
                             cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace")
        for line in p.stdout:
            _pytest_state["output"] += line
        _pytest_state.update(running=False, code=p.wait())

    threading.Thread(target=go, daemon=True).start()
    time.sleep(0.2)
    return _pytest_state


@app.get("/api/pytest")
def pytest_status():
    return _pytest_state


# ------------------------------------------------------------------ main
def main() -> None:
    import argparse

    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-ollama", action="store_true")
    a = ap.parse_args()
    _install_model_cache()
    threading.Thread(target=_worker, daemon=True).start()
    if not a.no_ollama and paths.OLLAMA_EXE.exists():
        threading.Thread(target=ollama.ensure_server, daemon=True).start()
    print(f"subalign 功能测试台: http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
