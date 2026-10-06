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
    if "subalign.tts.dubbing" in sys.modules:
        sys.modules["subalign.tts.dubbing"]._ASR.clear()       # QA / emphasis models of AI 配音
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
TASKS = ("align", "lyrics", "separate", "roughcut", "videocut", "studio", "convert", "translate", "proofread")
RECORDINGS = paths.WORK / "recordings"


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
                     sample: str = Form(""), recording: str = Form(""), bgm: Optional[UploadFile] = File(None)):
    if task not in TASKS:
        raise HTTPException(400, f"unknown task {task}")
    wd = _new_workdir()
    meta: Dict[str, Any] = {}
    argv: List[str] = [task]

    # --- primary input
    media_path: Optional[Path] = None
    if task in ("align", "lyrics", "separate", "roughcut", "videocut", "studio"):
        if media is not None and media.filename:
            media_path = await _save_upload(media, wd / "input")
        elif sample or recording:
            src = (paths.WORK / "samples" / Path(sample).name) if sample else (RECORDINGS / Path(recording).name)
            if not src.exists():
                raise HTTPException(400, "sample / recording not found")
            media_path = wd / "input" / src.name
            shutil.copy(src, media_path)
        else:
            raise HTTPException(400, "请上传音频/视频文件")
        argv.append(str(media_path))
        meta["media"] = f"/jobs/{wd.name}/file/input/{media_path.name}"
    if task == "studio" and bgm is not None and bgm.filename:
        (wd / "input" / "bgm").mkdir(exist_ok=True)
        bgm_path = await _save_upload(bgm, wd / "input" / "bgm")
        argv += ["--bgm", str(bgm_path)]
        meta["bgm"] = f"/jobs/{wd.name}/file/input/bgm/{bgm_path.name}"

    # --- secondary text input (script / lyrics / subtitle)
    text_path: Optional[Path] = None
    if text_file is not None and text_file.filename:
        text_path = await _save_upload(text_file, wd / "input")
    elif text_content.strip():
        name = Path(text_name or ("script.txt" if task in ("align", "lyrics") else "input.srt")).name
        text_path = wd / "input" / name
        text_path.write_text(text_content, encoding="utf-8")
    if task in ("align", "roughcut", "videocut") and text_path:
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
    if task in ("align", "lyrics", "convert", "roughcut", "videocut") and "-f" in argv:
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
            out.append({"names": names, "url": f"/fonts/sys/{fn}", "preview": f"/fonts/preview/sys/{fn}.png"})
    return out


@app.get("/fonts/preview/{kind}/{name}.png")
def font_preview(kind: str, name: str):
    """The sample text set in the font (for the font picker), rendered once and cached."""
    from subalign.fontlib import FontLibrary, preview_png, render_preview

    try:
        if kind == "free":
            lib = FontLibrary.load()
            e = next((x for x in lib.entries if x.id == name), None)
            if e is None:
                raise HTTPException(404)
            return FileResponse(preview_png(e), media_type="image/png")
        if kind == "sys" and name in {f for _, f in SYSTEM_FONTS} and (_FONT_DIR / name).exists():
            out = paths.WORK / "font_previews" / f"{name}.png"
            if not out.exists():
                render_preview(_FONT_DIR / name, out, "字幕预览 Subtitle 123")
            return FileResponse(out, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(404, f"preview failed: {e}")
    raise HTTPException(404)


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


@app.post("/api/jobs/{job_id}/videocut")
def videocut_apply(job_id: str, body: Dict[str, Any] = Body(...)):
    """Render a video rough cut again with reviewed decisions (no recognition).

    ``body.removals``: per removal of the plan ``{"mode": cut|mute|keep|null, "transition": auto|cut|fade|zoom}``
    (null mode = decide automatically); ``body.video``: optional picture settings.
    """
    from subalign.videocut import MODES, TRANSITIONS, VideoCutConfig

    parent = _load_job(job_id)
    if parent is None or parent.task != "videocut":
        raise HTTPException(404, "job not found")
    plan_file = next(iter(sorted((parent.workdir / "output").glob("*.videocut.json"))), None)
    if plan_file is None:
        raise HTTPException(400, "原任务没有剪辑计划")
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    edits = body.get("removals")
    if not isinstance(edits, list) or len(edits) != len(plan.get("removals", [])):
        raise HTTPException(400, "removals 与剪辑计划不匹配")
    for r, e in zip(plan["removals"], edits):
        e = e or {}
        mode = e.get("mode")
        r["locked"] = mode in MODES
        if mode in MODES:
            r["mode"] = mode
        r["transition"] = e.get("transition") if e.get("transition") in TRANSITIONS else "auto"
    video = plan.setdefault("settings", {}).setdefault("video", {})
    for k, v in (body.get("video") or {}).items():
        if k in VideoCutConfig.__dataclass_fields__ and v is not None:
            video[k] = v
    src_media = Path(parent.argv[1]) if len(parent.argv) > 1 else None
    if src_media is None or not src_media.exists():
        src_media = next((f for f in sorted((parent.workdir / "input").glob("*")) if f.suffix.lower() in MEDIA_EXT), None)
    if src_media is None:
        raise HTTPException(400, "原任务的视频已不存在")
    wd = _new_workdir()
    media = wd / "input" / src_media.name
    shutil.copy(src_media, media)
    pf = wd / "input" / "edited.videocut.json"
    pf.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    meta_p = parent.meta or {}
    # picture settings now come from the plan (its values win over the defaults only)
    drop = {"--min-shot", "--max-cuts", "--no-mute", "--mute-max", "--slide", "--transitions", "--zoom",
            "--fade-frames", "--content", "--llm", "--no-pauses", "--max-pause", "--keep-pause"}
    pairs = [p for p in meta_p.get("pairs", []) if p and p[0] not in drop]
    argv = _add_flags("videocut", ["videocut", str(media), "--plan", str(pf)], pairs, wd, "", meta_p.get("style_json", ""))
    meta = {k: v for k, v in meta_p.items() if k not in ("_command", "style_edit")}
    meta.update(pairs=pairs, parent=parent.id, edited=True, media=meta_p.get("media"))
    return _enqueue("videocut", argv, wd, meta)


@app.post("/api/recordings")
async def save_recording(file: UploadFile = File(...), name: str = Form("")):
    """Keep a take recorded in the browser (16/24-bit WAV) for processing."""
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    stem = "".join(c for c in (name or "录音") if c.isalnum() or c in "-_ ").strip()[:40] or "录音"
    dst = RECORDINGS / f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}.wav"
    with open(dst, "wb") as out:
        while True:
            chunk = await file.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return {"name": dst.name, "url": f"/recordings/{dst.name}", "size": dst.stat().st_size}


@app.get("/api/recordings")
def list_recordings():
    if not RECORDINGS.exists():
        return []
    return [{"name": f.name, "url": f"/recordings/{f.name}", "size": f.stat().st_size, "mtime": f.stat().st_mtime}
            for f in sorted(RECORDINGS.glob("*.wav"), key=lambda f: -f.stat().st_mtime)]


@app.get("/recordings/{name}")
def recording_file(name: str):
    p = RECORDINGS / Path(name).name
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p)


@app.post("/api/models/free")
def api_free():
    return {"freed": free_models()}


# ------------------------------------------------------------------ AI voice-over (AI 配音)
DUB_DIR = paths.WORK / "dubbing"
_tts_queue: "queue.Queue[tuple]" = queue.Queue()
_tts_state: Dict[str, Dict[str, Any]] = {}          # project id -> {"queued": [sid...], "running": sid, "error": str}
_tts_lock = threading.Lock()


def _dub_dir(pid: str) -> Path:
    d = DUB_DIR / Path(pid).name
    if not (d / "project.json").exists():
        raise HTTPException(404, "project not found")
    return d


def _tts_worker() -> None:
    from subalign.tts import dubbing

    while True:
        pid, kind, arg = _tts_queue.get()
        st = _tts_state.setdefault(pid, {"queued": [], "running": None})
        try:
            pdir = DUB_DIR / pid
            if kind == "synth":
                with _tts_lock:
                    if arg in st["queued"]:
                        st["queued"].remove(arg)
                    st["running"] = arg
                dubbing.synth_segment(pdir, arg)
            elif kind == "assemble":
                st["running"] = "mix"
                dubbing.assemble(pdir, dubbing.load(pdir))
            st.pop("error", None)
        except Exception as e:
            log.exception("tts %s %s failed", kind, arg)
            st["error"] = f"{type(e).__name__}: {e}"
        finally:
            st["running"] = None
            if not st["queued"] and kind == "synth" and st.get("auto_mix"):
                st["auto_mix"] = False
                _tts_queue.put((pid, "assemble", None))


def _tts_enqueue(pid: str, sids: List[int], mix: bool = True) -> None:
    with _tts_lock:
        st = _tts_state.setdefault(pid, {"queued": [], "running": None})
        for sid in sids:
            if sid not in st["queued"] and st["running"] != sid:
                st["queued"].append(sid)
                _tts_queue.put((pid, "synth", sid))
        st["auto_mix"] = mix or st.get("auto_mix", False)


def _dub_view(pid: str) -> Dict[str, Any]:
    from subalign.tts import dubbing

    proj = dubbing.load(DUB_DIR / pid)
    st = _tts_state.get(pid, {})
    queued = set(st.get("queued", []))
    for s in proj["segments"]:
        if s["id"] in queued:
            s["status"] = "queued"
        elif s.get("status") == "running" and st.get("running") != s["id"]:
            s["status"] = "error" if not s.get("audio") else "done"     # interrupted by a restart
        s["url"] = f"/dub/{pid}/{s['audio']}?v={s.get('updated', 0)}" if s.get("audio") else None
        for t in s.get("takes", []):
            t["url"] = f"/dub/{pid}/{t['file']}"
    if proj.get("mix"):
        proj["mix"]["url"] = f"/dub/{pid}/{proj['mix']['file']}?v={proj['mix']['built']}"
    proj.update(id=pid, busy=bool(st.get("queued")) or st.get("running") is not None, running=st.get("running"),
                error=st.get("error"), spk_url=f"/dub/{pid}/ref/{Path(proj['spk']).name}",
                emo_url=f"/dub/{pid}/ref/{Path(proj['emo']).name}" if proj.get("emo") else None)
    return proj


@app.get("/api/tts/status")
def tts_status():
    from subalign.tts.dubbing import WORKER

    return {"available": WORKER.available(), "running": bool(WORKER.proc and WORKER.proc.poll() is None),
            "python": str(WORKER.python()), "model": str(ROOT / "models" / "indextts-2.5")}


@app.post("/api/tts/prepare")
def tts_prepare(body: Dict[str, Any] = Body(...)):
    """One-click script clean-up: notes / emoji removed, numbers read out, sentences split,
    polyphonic characters listed for review."""
    from subalign.tts import textprep

    return textprep.prepare(body.get("text", ""), remove_notes=body.get("remove_notes", True),
                            max_chars=int(body.get("max_chars") or 60), lang=body.get("lang") or "zh")


async def _ref_audio(dst: Path, upload: Optional[UploadFile], recording: str, prev: Optional[Path] = None) -> Optional[Path]:
    dst.mkdir(parents=True, exist_ok=True)
    if upload is not None and upload.filename:
        return await _save_upload(upload, dst)
    if recording:
        src = RECORDINGS / Path(recording).name
        if not src.exists():
            raise HTTPException(400, "recording not found")
        shutil.copy(src, dst / src.name)
        return dst / src.name
    return prev


def _trim_reference(p: Path, max_s: float = 15.0) -> Path:
    """IndexTTS only uses ~15 s of a reference; long uploads are cut (and converted to wav)."""
    from subalign.audio.io import load_audio, save_audio

    y = load_audio(p, 24000)
    if p.suffix.lower() == ".wav" and len(y) <= max_s * 24000:
        return p
    out = p.with_name(p.stem + ".ref.wav")
    save_audio(out, y[:int(max_s * 24000)], 24000)
    return out


@app.post("/api/tts/projects")
async def tts_create(script: str = Form(...), title: str = Form(""), config: str = Form("{}"),
                     spk: Optional[UploadFile] = File(None), spk_recording: str = Form(""),
                     emo: Optional[UploadFile] = File(None), emo_recording: str = Form(""), start: bool = Form(True)):
    from subalign.tts import dubbing

    if not script.strip():
        raise HTTPException(400, "请输入文稿")
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    pdir = DUB_DIR / pid
    spk_p = await _ref_audio(pdir / "ref", spk, spk_recording)
    if spk_p is None:
        raise HTTPException(400, "请上传或录制音色参考音频 A")
    emo_p = await _ref_audio(pdir / "ref", emo, emo_recording)
    spk_p, emo_p = _trim_reference(spk_p), _trim_reference(emo_p) if emo_p else None
    try:
        cfg = dubbing.DubConfig(**{k: v for k, v in json.loads(config).items() if k in dubbing.DubConfig.__dataclass_fields__})
    except (json.JSONDecodeError, TypeError) as e:
        raise HTTPException(400, f"config: {e}")
    proj = dubbing.create_project(pdir, script, spk_p, emo_p, cfg, title)
    if start:
        _tts_enqueue(pid, [s["id"] for s in proj["segments"]])
    return _dub_view(pid)


@app.get("/api/tts/projects")
def tts_list():
    out = []
    for d in sorted(DUB_DIR.glob("*/project.json"), reverse=True) if DUB_DIR.exists() else []:
        try:
            p = json.loads(d.read_text(encoding="utf-8"))
        except Exception:
            continue
        segs = p.get("segments", [])
        out.append({"id": d.parent.name, "title": p.get("title"), "created": p.get("created"), "n": len(segs),
                    "done": sum(1 for s in segs if s.get("audio")), "mix": bool(p.get("mix"))})
    return out


@app.get("/api/tts/projects/{pid}")
def tts_get(pid: str):
    _dub_dir(pid)
    return _dub_view(pid)


@app.post("/api/tts/projects/{pid}/synth")
def tts_synth(pid: str, body: Dict[str, Any] = Body({})):
    """Generate sentences: ``ids`` (default: every sentence without a current take)."""
    from subalign.tts import dubbing

    pdir = _dub_dir(pid)
    proj = dubbing.load(pdir)
    ids = body.get("ids") or [s["id"] for s in proj["segments"] if not s.get("audio") or s.get("status") in ("edited", "error")]
    _tts_enqueue(pid, [int(i) for i in ids], mix=body.get("mix", True))
    return _dub_view(pid)


@app.post("/api/tts/projects/{pid}/segments/{sid}")
def tts_edit_segment(pid: str, sid: int, body: Dict[str, Any] = Body(...)):
    """Edit a sentence (text / pause after it), optionally regenerate it right away;
    ``take`` picks an earlier take instead."""
    from subalign.tts import dubbing

    pdir = _dub_dir(pid)
    try:
        if body.get("take"):
            dubbing.use_take(pdir, sid, body["take"])
            _tts_queue.put((pid, "assemble", None))
        else:
            dubbing.edit_segment(pdir, sid, body.get("text"), body.get("pause_after"))
    except (KeyError, StopIteration):
        raise HTTPException(404, "sentence / take not found")
    if body.get("regenerate"):
        _tts_enqueue(pid, [sid])
    return _dub_view(pid)


@app.post("/api/tts/projects/{pid}/config")
def tts_config(pid: str, body: Dict[str, Any] = Body(...)):
    from subalign.tts import dubbing

    dubbing.set_config(_dub_dir(pid), **body)
    return _dub_view(pid)


@app.post("/api/tts/projects/{pid}/assemble")
def tts_assemble(pid: str):
    _dub_dir(pid)
    _tts_queue.put((pid, "assemble", None))
    _tts_state.setdefault(pid, {"queued": [], "running": None})["running"] = "mix"
    return _dub_view(pid)


@app.post("/api/tts/projects/{pid}/export")
def tts_export(pid: str, body: Dict[str, Any] = Body({})):
    from subalign.tts import dubbing

    pdir = _dub_dir(pid)
    if not (pdir / "mix.wav").exists():
        raise HTTPException(400, "还没有合成总音频")
    fmt = body.get("format", "mp3")
    if fmt not in ("wav", "mp3", "aac", "flac", "opus"):
        raise HTTPException(400, "unknown format")
    p = dubbing.export_mix(pdir, fmt, body.get("bitrate", "320k"), int(body.get("bit_depth", 24)),
                           int(body.get("sample_rate", 48000)), int(body.get("channels", 1)))
    return {"url": f"/dub/{pid}/{p.name}?v={int(time.time())}", "name": p.name, "size": p.stat().st_size}


@app.post("/api/tts/projects/{pid}/to-studio")
def tts_to_studio(pid: str):
    """Hand the assembled voice to the voice-over chain (it appears among the recordings)."""
    from subalign.tts import dubbing

    pdir = _dub_dir(pid)
    if not (pdir / "mix.wav").exists():
        raise HTTPException(400, "还没有合成总音频")
    proj = dubbing.load(pdir)
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    stem = "".join(c for c in (proj.get("title") or "AI配音") if c.isalnum() or c in "-_ ").strip()[:30] or "AI配音"
    dst = RECORDINGS / f"{time.strftime('%Y%m%d-%H%M%S')}-AI-{stem}.wav"
    shutil.copy(pdir / "mix.wav", dst)
    return {"name": dst.name}


@app.post("/api/tts/worker/{action}")
def tts_worker_ctl(action: str):
    from subalign.tts.dubbing import WORKER

    if action == "stop":
        WORKER.stop()
    elif action == "start":
        WORKER.request(cmd="load")
    else:
        raise HTTPException(404)
    return tts_status()


@app.get("/dub/{pid}/{rel:path}")
def dub_file(pid: str, rel: str):
    base = _dub_dir(pid).resolve()
    p = (base / rel).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


# ------------------------------------------------------------------ dubbing translation (音频翻译)
# source (recording / video -> recognition, or a subtitle / script) -> translation that keeps
# the length and where the nouns are heard -> AI voice-over on the original timeline
DT_DIR = paths.WORK / "dubtrans"
_dt_lock = threading.RLock()
_dt_state: Dict[str, Dict[str, Any]] = {}           # project id -> {"busy", "stage", "done", "total", "error"}
_dt_live: Dict[str, Dict[str, Any]] = {}            # projects being translated (edits go to the same object)
TTS_LANG = {"zh": "ZH", "zh-cn": "ZH", "zh-tw": "ZH", "zh-hant": "ZH", "yue": "ZH", "en": "EN", "ja": "JA", "es": "ES"}


def _dt_dir(pid: str) -> Path:
    d = DT_DIR / Path(pid).name
    if not (d / "project.json").exists():
        raise HTTPException(404, "project not found")
    return d


def _dt_get(pid: str) -> Dict[str, Any]:
    if pid in _dt_live:
        return _dt_live[pid]
    return json.loads((_dt_dir(pid) / "project.json").read_text(encoding="utf-8"))


def _dt_save(pid: str, proj: Dict[str, Any]) -> None:
    with _dt_lock:
        d = DT_DIR / pid
        tmp = d / "project.json.tmp"
        tmp.write_text(json.dumps(proj, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(d / "project.json")


def _dt_cfg(proj: Dict[str, Any]):
    from subalign.translate.isochrony import IsoConfig

    return IsoConfig(**{k: v for k, v in proj["config"].items() if k in IsoConfig.__dataclass_fields__})


def _llm_client(llm: Dict[str, Any]):
    """The page's LLM settings (the key is never written to disk); some randomness so
    that the candidates differ."""
    from subalign.llm import make_client

    prov = (llm or {}).get("provider") or "ollama"
    if prov == "ollama":
        ollama.ensure_server()
    return make_client(prov, llm.get("model") or None, llm.get("base_url") or None, llm.get("api_key") or None,
                       temperature=0.7)


def _job_document(job: Job):
    """The recognised document (json output) of an alignment job."""
    from subalign.models import Document

    out = job.workdir / "output"
    media = Path(job.argv[1]).stem if len(job.argv) > 1 else ""
    cands = [out / f"{media}.json"] + sorted(out.glob("*.json"))
    for p in cands:
        if p.exists() and not any(p.name.endswith(x) for x in (".proofread.json", ".studio.json", ".roughcut.json")):
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict) and "lines" in d:
                return Document.from_dict(d)
    raise RuntimeError("识别结果里没有 json 文档")


def _dt_run(pid: str, llm: Dict[str, Any], ids: Optional[List[int]] = None, hints: Optional[Dict[int, str]] = None) -> None:
    from subalign.translate import isochrony

    st = _dt_state.setdefault(pid, {})
    st.update(busy=True, stage="start", done=0, total=0, error=None, t0=time.time(), stage_t0=time.time(), stop=False,
              stopped=False)
    try:
        with _dt_lock:
            proj = _dt_live[pid] = _dt_get(pid)
        src = proj["source"]
        if src.get("job") and not proj["sentences"]:
            st["stage"] = "transcribe"
            while True:
                job = _load_job(src["job"])
                if job is None:
                    raise RuntimeError("识别任务不存在")
                if job.status in ("done", "error"):
                    break
                time.sleep(1.0)
            if job.status == "error":
                raise RuntimeError(f"语音识别失败：{job.error}")
            doc = _job_document(job)
            with _dt_lock:
                proj["sentences"] = [] if proj.get("merge", True) else isochrony.sentences_from_document(doc, merge=False)
                (_dt_dir(pid) / "input").mkdir(exist_ok=True)
                (_dt_dir(pid) / "input" / "cues.json").write_text(json.dumps(doc.to_dict(), ensure_ascii=False), encoding="utf-8")
                proj["source"]["cues"] = "input/cues.json"
                proj["source"]["media_path"] = job.argv[1] if len(job.argv) > 1 else None
                vocals = sorted((job.workdir / "output" / "work").glob("*vocals*.wav")) if (job.workdir / "output" / "work").exists() else []
                proj["source"]["vocals_path"] = str(vocals[0]) if vocals else None
                _dt_save(pid, proj)
        cfg = _dt_cfg(proj)
        if ((llm or {}).get("provider") or "ollama") == "ollama":
            _gpu_for_local_llm()
        client = _llm_client(llm)
        if not proj["sentences"] and src.get("cues"):
            # cue boundaries are only timing: whole sentences again (LLM when there is no punctuation)
            from subalign.models import Document

            st["stage"], st["stage_t0"] = "segment", time.time()
            doc = Document.from_dict(json.loads((_dt_dir(pid) / src["cues"]).read_text(encoding="utf-8")))
            sents = isochrony.resegment(doc, cfg.source, client)
            with _dt_lock:
                proj["sentences"] = sents
                _dt_save(pid, proj)
        if not proj["sentences"]:
            raise RuntimeError("没有可翻译的句子")

        st["plan"] = (["segment"] if src.get("cues") else []) + ["anchors", "translate"] + \
            [f"refine{r}" for r in range(1, cfg.rounds + 1)] + (["judge"] if cfg.judge else [])

        def progress(stage: str, done: int, total: int) -> None:
            if stage != st.get("stage"):
                st["stage_t0"] = time.time()
            st.update(stage=stage, done=done, total=total)
            _dt_save(pid, proj)
        isochrony.translate(proj["sentences"], client, cfg, ids=ids, hints=hints, progress=progress,
                            stop=lambda: bool(st.get("stop")))
        st["stopped"] = bool(st.get("stop"))
        st["stage"] = "done"
        _dt_save(pid, proj)
    except Exception as e:
        log.exception("dub translation %s failed", pid)
        st["error"] = f"{type(e).__name__}: {e}"
    finally:
        with _dt_lock:
            _dt_live.pop(pid, None)
        st["busy"] = False


def _gpu_for_local_llm() -> None:
    """A local model needs ~5 GB of VRAM: drop the cached recognition models and stop an
    idle TTS worker (both reload on demand) - with them loaded Ollama fails with
    'cudaMalloc failed: out of memory' on an 11 GB card."""
    from subalign.tts.dubbing import WORKER

    free_models()
    busy = any(st.get("queued") or st.get("running") is not None for st in _tts_state.values())
    if not busy and WORKER.proc and WORKER.proc.poll() is None:
        log.info("stopping the idle TTS worker to make room for the local LLM")
        WORKER.stop()


def _dt_start(pid: str, llm: Dict[str, Any], ids: Optional[List[int]] = None, hints: Optional[Dict[int, str]] = None) -> None:
    if _dt_state.get(pid, {}).get("busy"):
        raise HTTPException(409, "正在翻译，请稍候")
    _dt_state[pid] = {"busy": True, "stage": "start", "done": 0, "total": 0, "error": None}
    threading.Thread(target=_dt_run, args=(pid, llm, ids, hints), daemon=True).start()


def _dt_view(pid: str) -> Dict[str, Any]:
    proj = dict(_dt_get(pid))
    st = _dt_state.get(pid, {})
    src = proj.get("source") or {}
    media = src.get("media_url")
    if src.get("job"):
        job = _load_job(src["job"])
        media = job.meta.get("media") if job else None
        if job and job.status in ("queued", "running") and not proj.get("sentences"):
            st = dict(st, job_status=job.status, job_log=job.log[-3:])
    now = time.time()
    proj["progress"] = {"plan": st.get("plan") or [], "elapsed": round(now - st["t0"], 1) if st.get("t0") else None,
                        "stage_elapsed": round(now - st["stage_t0"], 1) if st.get("stage_t0") else None,
                        "stopping": bool(st.get("stop")) and bool(st.get("busy")), "stopped": bool(st.get("stopped"))}
    proj.update(id=pid, busy=bool(st.get("busy")), stage=st.get("stage"), done=st.get("done"), total=st.get("total"),
                error=st.get("error"), job_status=st.get("job_status"), job_log=st.get("job_log"), media_url=media,
                has_voice=bool(src.get("media_path")))
    vinfo = _dt_video_info(src.get("media_path"))
    proj["video_info"] = vinfo
    proj["is_video"] = bool(vinfo)
    proj["dub_info"] = [d for d in (_dub_summary(x) for x in proj.get("dubbing") or []) if d]
    vs = []
    for v in proj.get("videos") or []:
        v = dict(v, **(_dv_state.get(v["id"]) or {}))
        if v.get("status") in ("queued", "running") and v["id"] not in _dv_state:
            v.update(status="error", error="服务重启，任务已中断，请重新生成")
        v["urls"] = {k: f"/dt/{pid}/video/{v['id']}/{name}" for k, name in (v.get("files") or {}).items()}
        vs.append(v)
    proj["videos"] = vs
    return proj


_video_info_cache: Dict[str, Optional[Dict[str, Any]]] = {}


def _dt_video_info(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Source video facts (None for audio-only sources), cached per file."""
    if not path or not Path(path).exists():
        return None
    if path not in _video_info_cache:
        try:
            from subalign.dubvideo import probe

            i = probe(Path(path))
            _video_info_cache[path] = {"width": i["width"], "height": i["height"], "fps": round(float(i["fps"]), 3),
                                       "codec": i["codec"], "bit_rate": i["bit_rate"], "container": i["ext"],
                                       "duration": round(i["duration"], 2)}
        except Exception:
            _video_info_cache[path] = None
    return _video_info_cache[path]


def _dub_summary(did: str) -> Optional[Dict[str, Any]]:
    f = DUB_DIR / Path(did).name / "project.json"
    if not f.exists():
        return None
    try:
        p = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None
    segs = p.get("segments", [])
    return {"id": did, "title": p.get("title"), "n": len(segs), "done": sum(1 for s in segs if s.get("audio")),
            "mix": bool(p.get("mix")), "timeline": bool(p.get("config", {}).get("timeline")),
            "built": (p.get("mix") or {}).get("built")}


@app.post("/api/dubtrans/projects")
async def dt_create(source: str = Form("text"), media: Optional[UploadFile] = File(None), recording: str = Form(""),
                    job: str = Form(""), text_content: str = Form(""), text_name: str = Form(""),
                    src_lang: str = Form("zh"), tgt_lang: str = Form("en"), asr_model: str = Form(""),
                    config: str = Form("{}"), llm: str = Form("{}"), title: str = Form(""), merge: bool = Form(True)):
    from subalign.translate import isochrony

    try:
        cfg_d, llm_d = json.loads(config), json.loads(llm)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"JSON: {e}")
    cfg = isochrony.IsoConfig(**{k: v for k, v in cfg_d.items() if k in isochrony.IsoConfig.__dataclass_fields__})
    cfg.source, cfg.target = src_lang, tgt_lang
    if isochrony.base_lang(src_lang) == isochrony.base_lang(tgt_lang):
        raise HTTPException(400, "源语言和目标语言相同")
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    pdir = DT_DIR / pid
    pdir.mkdir(parents=True)
    src: Dict[str, Any] = {"kind": source}
    sents: List[Dict[str, Any]] = []
    if source == "media":
        wd = _new_workdir()
        if media is not None and media.filename:
            mp = await _save_upload(media, wd / "input")
        elif recording:
            rp = RECORDINGS / Path(recording).name
            if not rp.exists():
                raise HTTPException(400, "recording not found")
            mp = wd / "input" / rp.name
            shutil.copy(rp, mp)
        else:
            raise HTTPException(400, "请上传音频 / 视频")
        pairs = [["--lang", isochrony.base_lang(src_lang)], ["--mode", "speech"], ["-f", "srt,json"]]
        if asr_model:
            pairs.append(["--asr-model", asr_model])
        argv = _add_flags("align", ["align", str(mp)], pairs, wd, "", "")
        jid = _enqueue("align", argv, wd, {"media": f"/jobs/{wd.name}/file/input/{mp.name}", "pairs": pairs,
                                           "glossary": "", "style_json": "", "purpose": "音频翻译"})["id"]
        src.update(job=jid, name=mp.name)
    elif source == "job":
        j = _load_job(job)
        if j is None or j.task not in ("align", "lyrics"):
            raise HTTPException(400, "对齐任务不存在")
        src.update(job=j.id, name=Path(j.argv[1]).name if len(j.argv) > 1 else j.id)
    else:
        if not text_content.strip():
            raise HTTPException(400, "请粘贴或上传字幕 / 文稿")
        name = Path(text_name or "script.txt").name
        ext = Path(name).suffix.lower()
        (pdir / "input").mkdir()
        tp = pdir / "input" / name
        tp.write_text(text_content, encoding="utf-8")
        if ext in (".srt", ".vtt", ".ass", ".ssa", ".lrc", ".json", ".sbv", ".ttml", ".qrc", ".yrc"):
            from subalign.formats import read

            try:
                doc = read(tp)
            except Exception as e:
                raise HTTPException(400, f"字幕解析失败：{e}")
            if not doc.lines:
                raise HTTPException(400, "字幕里没有内容")
            if merge:
                # re-cut into whole sentences in the background (may need the LLM)
                (pdir / "input" / "cues.json").write_text(json.dumps(doc.to_dict(), ensure_ascii=False), encoding="utf-8")
                src["cues"] = "input/cues.json"
                sents = []
            else:
                sents = isochrony.sentences_from_document(doc, merge=False)
        else:
            sents = isochrony.sentences_from_text(text_content, src_lang)
        src.update(name=name)
        # the video / recording the subtitles belong to (optional): original voice + translated video
        mp = await _dt_media(pdir, media, recording)
        if mp is not None:
            src.update(media_path=str(mp), media_url=f"/dt/{pid}/input/media/{mp.name}")
    proj = {"version": 1, "title": title or (src.get("name") or "音频翻译"), "created": time.time(), "merge": merge,
            "config": {k: getattr(cfg, k) for k in isochrony.IsoConfig.__dataclass_fields__}, "source": src,
            "sentences": sents, "llm": {k: llm_d.get(k, "") for k in ("provider", "model", "base_url")}, "dubbing": []}
    _dt_save(pid, proj)
    _dt_start(pid, llm_d)
    return _dt_view(pid)


async def _dt_media(pdir: Path, media: Optional[UploadFile], recording: str) -> Optional[Path]:
    dst = pdir / "input" / "media"
    if media is not None and media.filename:
        dst.mkdir(parents=True, exist_ok=True)
        for old in dst.glob("*"):
            old.unlink()
        return await _save_upload(media, dst)
    if recording:
        rp = RECORDINGS / Path(recording).name
        if not rp.exists():
            raise HTTPException(400, "recording not found")
        dst.mkdir(parents=True, exist_ok=True)
        for old in dst.glob("*"):
            old.unlink()
        shutil.copy(rp, dst / rp.name)
        return dst / rp.name
    return None


@app.post("/api/dubtrans/projects/{pid}/media")
async def dt_attach_media(pid: str, media: Optional[UploadFile] = File(None), recording: str = Form("")):
    """Attach the video / recording a subtitle- or script-based translation belongs to
    (its timing must match the subtitles): original voice for cloning + translated video."""
    pdir = _dt_dir(pid)
    with _dt_lock:
        proj = _dt_get(pid)
        if proj["source"].get("job"):
            raise HTTPException(400, "这个项目来自语音识别，已经有原视频 / 音频")
    mp = await _dt_media(pdir, media, recording)
    if mp is None:
        raise HTTPException(400, "请选择视频或音频文件")
    with _dt_lock:
        proj = _dt_get(pid)
        proj["source"].update(media_path=str(mp), media_url=f"/dt/{pid}/input/media/{mp.name}")
        _dt_save(pid, proj)
    return _dt_view(pid)


@app.get("/api/dubtrans/projects")
def dt_list():
    out = []
    for d in sorted(DT_DIR.glob("*/project.json"), reverse=True) if DT_DIR.exists() else []:
        try:
            p = json.loads(d.read_text(encoding="utf-8"))
        except Exception:
            continue
        ss = p.get("sentences", [])
        out.append({"id": d.parent.name, "title": p.get("title"), "created": p.get("created"), "n": len(ss),
                    "src": p["config"].get("source"), "tgt": p["config"].get("target"),
                    "done": sum(1 for s in ss if s.get("translation")), "ok": sum(1 for s in ss if _dt_ok(s))})
    return out


def _dt_ok(s: Dict[str, Any]) -> bool:
    c = s.get("candidates") or []
    i = s.get("choice")
    return i is not None and 0 <= i < len(c) and bool(c[i].get("ok"))


@app.get("/api/dubtrans/projects/{pid}")
def dt_get(pid: str):
    _dt_get(pid)
    return _dt_view(pid)


@app.post("/api/dubtrans/projects/{pid}/sentences/{sid}")
def dt_sentence(pid: str, sid: int, body: Dict[str, Any] = Body(...)):
    """``choice``: use candidate i; ``text``: own translation; ``regenerate`` (+ ``hint``):
    more candidates for this sentence."""
    from subalign.translate import isochrony

    if body.get("regenerate"):
        with _dt_lock:
            proj = _dt_get(pid)
            s = next((x for x in proj["sentences"] if x["id"] == sid), None)
            if s is None:
                raise HTTPException(404, "sentence not found")
            s["locked"] = False
            _dt_save(pid, proj)
        hint = (body.get("hint") or "").strip()
        _dt_start(pid, body.get("llm") or {}, [sid], {sid: hint} if hint else None)
        return _dt_view(pid)
    with _dt_lock:
        proj = _dt_get(pid)
        s = next((x for x in proj["sentences"] if x["id"] == sid), None)
        if s is None:
            raise HTTPException(404, "sentence not found")
        if "choice" in body:
            i = int(body["choice"])
            if not 0 <= i < len(s.get("candidates") or []):
                raise HTTPException(400, "no such candidate")
            s["choice"], s["locked"], s["translation"] = i, True, s["candidates"][i]["text"]
        elif body.get("text", "").strip():
            if "target" not in s:
                raise HTTPException(400, "这一句还没有分析完")
            isochrony.set_custom(s, body["text"], _dt_cfg(proj))
        elif body.get("auto"):
            s["locked"] = False
            isochrony.rank(s)
        _dt_save(pid, proj)
    return _dt_view(pid)


@app.post("/api/dubtrans/projects/{pid}/stop")
def dt_stop(pid: str):
    """Stop after the current batch; everything translated so far is kept."""
    st = _dt_state.get(pid)
    if not st or not st.get("busy"):
        raise HTTPException(409, "没有在运行")
    st["stop"] = True
    return _dt_view(pid)


@app.post("/api/dubtrans/projects/{pid}/retranslate")
def dt_retranslate(pid: str, body: Dict[str, Any] = Body({})):
    """Translate again: ``ids`` (default all); ``reset`` drops the earlier candidates."""
    with _dt_lock:
        proj = _dt_get(pid)
        ids = body.get("ids")
        for s in proj["sentences"]:
            if ids is None or s["id"] in ids:
                s["locked"] = False
                if body.get("reset"):
                    s.update(candidates=[], choice=None, translation=None)
        _dt_save(pid, proj)
    _dt_start(pid, body.get("llm") or {}, ids)
    return _dt_view(pid)


@app.post("/api/dubtrans/projects/{pid}/config")
def dt_config(pid: str, body: Dict[str, Any] = Body(...)):
    """Tolerance / candidates / rounds / judge / ratio / instructions; budgets and scores
    are recomputed locally (no new translation)."""
    from subalign.translate import isochrony

    with _dt_lock:
        proj = _dt_get(pid)
        for k, v in body.items():
            if k in isochrony.IsoConfig.__dataclass_fields__ and k not in ("source", "target"):
                proj["config"][k] = v
        cfg = _dt_cfg(proj)
        for s in proj["sentences"]:
            if "target" in s:
                isochrony.remeasure(s, cfg)
        _dt_save(pid, proj)
    return _dt_view(pid)


@app.get("/api/dubtrans/projects/{pid}/export")
def dt_export(pid: str, fmt: str = "srt"):
    from fastapi.responses import Response
    from subalign.translate import isochrony

    proj = _dt_get(pid)
    ss = proj["sentences"]
    tgt = proj["config"].get("target", "tr")
    if fmt in ("srt", "srt2"):
        if not any(s.get("start") is not None for s in ss):
            raise HTTPException(400, "文稿没有时间轴，请导出 txt")
        # whole sentences can be long: broken into readable cues like the burned-in subtitles
        from subalign.dubvideo import subtitle_document
        from subalign.formats import write
        from subalign.segment.layout import get_layout
        from subalign.segment.linebreak import segment_document

        vi = _dt_video_info(proj["source"].get("media_path")) or {"width": 1920, "height": 1080}
        doc = subtitle_document([{"text": s.get("translation") or "", "original": s["text"], "start": s.get("start"),
                                  "end": s.get("end")} for s in ss], bilingual=fmt == "srt2", language=tgt)
        doc = segment_document(doc, get_layout(f"{vi['width']}x{vi['height']}"))
        from subalign.dubvideo import clean_srt

        body, name = clean_srt(write(doc, "srt", bilingual=fmt == "srt2")), f"{pid}.{tgt}.srt"
    elif fmt == "txt":
        body, name = "\n".join(s.get("translation") or "" for s in ss) + "\n", f"{pid}.{tgt}.txt"
    elif fmt == "json":
        body, name = json.dumps(proj, ensure_ascii=False, indent=1), f"{pid}.json"
    else:
        raise HTTPException(400, "unknown format")
    return Response(body, media_type="text/plain; charset=utf-8",
                    headers={"Content-Disposition": f"attachment; filename*=UTF-8''{name}"})


def _source_voice(proj: Dict[str, Any], dst: Path) -> Path:
    """A voice reference cut from the original recording (its vocals stem when the
    recognition separated one): the longest stretch of continuous speech, 5-15 s."""
    from subalign.audio.io import load_audio, save_audio

    src = proj["source"]
    media = src.get("vocals_path") or src.get("media_path")
    if not media or not Path(media).exists():
        raise HTTPException(400, "没有原始音频，无法使用原声音色")
    ss = [s for s in proj["sentences"] if s.get("start") is not None and s.get("end") is not None]
    best, best_len = None, 0.0
    for i in range(len(ss)):                         # contiguous runs of sentences, at most 15 s
        a, b = ss[i]["start"], ss[i]["end"]
        for j in range(i + 1, len(ss)):
            if ss[j]["start"] - b > 0.8 or ss[j]["end"] - a > 15:
                break
            b = ss[j]["end"]
        if b - a > best_len:
            best, best_len = (a, b), b - a
        if best_len >= 12:
            break
    a, b = best or (0.0, 12.0)
    y = load_audio(media, 24000)
    seg = y[int(max(0.0, a - 0.05) * 24000):int(min(b + 0.15, a + 15) * 24000)]
    if len(seg) < 24000 * 2:
        raise HTTPException(400, "原音频里找不到足够长的连续人声")
    dst.mkdir(parents=True, exist_ok=True)
    out = dst / "source_voice.wav"
    save_audio(out, seg, 24000)
    return out


@app.post("/api/dubtrans/projects/{pid}/to-dubbing")
async def dt_to_dubbing(pid: str, use_source: bool = Form(True), spk: Optional[UploadFile] = File(None),
                        spk_recording: str = Form(""), emo: Optional[UploadFile] = File(None), emo_recording: str = Form(""),
                        config: str = Form("{}"), title: str = Form(""), timeline: bool = Form(True), start: bool = Form(True)):
    """The translation -> an AI voice-over project (one sentence each, original timing kept)."""
    from subalign.tts import dubbing

    proj = _dt_get(pid)
    if _dt_state.get(pid, {}).get("busy"):
        raise HTTPException(409, "还在翻译，请等翻译完成")
    sents = [{"text": s.get("translation") or "", "start": s.get("start"), "end": s.get("end"), "source_text": s["text"],
              "dt_id": s["id"]}
             for s in proj["sentences"] if (s.get("translation") or "").strip()]
    if not sents:
        raise HTTPException(400, "还没有译文")
    try:
        cd = json.loads(config)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"config: {e}")
    did = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    ddir = DUB_DIR / did
    try:
        spk_p = await _ref_audio(ddir / "ref", spk, spk_recording)
        if spk_p is None:
            if not use_source:
                raise HTTPException(400, "请上传或选择音色参考 A")
            spk_p = _source_voice(proj, ddir / "ref")
        emo_p = await _ref_audio(ddir / "ref", emo, emo_recording)
        spk_p, emo_p = _trim_reference(spk_p), _trim_reference(emo_p) if emo_p else None
    except BaseException:
        shutil.rmtree(ddir, ignore_errors=True)         # no half-made project in the list
        raise
    tgt = proj["config"].get("target", "en").lower()
    cd.update(lang=TTS_LANG.get(tgt, tgt.split("-")[0].upper()),
              timeline=bool(timeline and all(s["start"] is not None for s in sents)))
    cfg = dubbing.DubConfig(**{k: v for k, v in cd.items() if k in dubbing.DubConfig.__dataclass_fields__})
    dp = dubbing.create_project(ddir, "", spk_p, emo_p, cfg, title or proj.get("title", ""), sentences=sents,
                                source={"dubtrans": pid, "media_url": _dt_view(pid).get("media_url")})
    with _dt_lock:
        p2 = _dt_get(pid)
        p2.setdefault("dubbing", []).append(did)
        _dt_save(pid, p2)
    if start:
        ollama.unload_all()                         # the translation model would crowd out IndexTTS
        _tts_enqueue(did, [s["id"] for s in dp["segments"]])
    return {"id": did}


# ------------------------------------------------------------------ translated video (视频翻译)
_dv_queue: "queue.Queue[tuple]" = queue.Queue()
_dv_state: Dict[str, Dict[str, Any]] = {}            # video id -> {"status", "stage", "progress", "error"}


def _dv_sentences(proj: Dict[str, Any], did: Optional[str]) -> List[Dict[str, Any]]:
    """Subtitle lines: the translation, timed where the dubbed sentence is actually
    spoken (mix timing); without a mix, at the original sentence times."""
    by_id = {s["id"]: s for s in proj["sentences"]}
    if did:
        f = DUB_DIR / Path(did).name / "project.json"
        if f.exists():
            dp = json.loads(f.read_text(encoding="utf-8"))
            timing = {t["id"]: t for t in (dp.get("mix") or {}).get("timing", [])}
            out = []
            for seg in dp.get("segments", []):
                t = timing.get(seg["id"])
                if not t:
                    continue
                src = by_id.get(seg.get("dt_id"))
                out.append({"text": (src or {}).get("translation") or seg["text"],
                            "original": (src or {}).get("text") or seg.get("source_text"),
                            "start": t["start"], "end": t["end"]})
            if out:
                return out
    return [{"text": s.get("translation") or "", "original": s["text"], "start": s.get("start"), "end": s.get("end")}
            for s in proj["sentences"] if s.get("translation") and s.get("start") is not None]


def _default_style_spec(proj: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Default subtitle style of a translation: free-for-commercial-use fonts for the
    target language (main row) and the source language (second row)."""
    from subalign.fontlib import FontLibrary

    spec = dict(preset_style("default"), font_size=None)
    lib = FontLibrary.load()
    cfg = (proj or {}).get("config") or {}
    for key, lang in (("main", cfg.get("target", "zh")), ("translation", cfg.get("source", "zh"))):
        e = lib.default_for(lang)
        if e:
            spec[key] = dict(spec[key], fontname=e.family)
    return spec


def _dv_ass(pid: str, did: Optional[str], bilingual: bool, spec: Dict[str, Any]) -> str:
    from subalign.dubvideo import subtitle_document
    from subalign.formats.ass import write_ass
    from subalign.segment.layout import get_layout
    from subalign.segment.linebreak import segment_document

    proj = _dt_get(pid)
    vi = _dt_video_info(proj["source"].get("media_path")) or {"width": 1920, "height": 1080}
    doc = subtitle_document(_dv_sentences(proj, did), bilingual, proj["config"].get("target", ""))
    layout = get_layout(f"{vi['width']}x{vi['height']}", int(spec["font_size"]) if spec.get("font_size") else None)
    from subalign.dubvideo import clean_ass
    from subalign.fontlib import FontLibrary

    text = clean_ass(write_ass(segment_document(doc, layout), style=_style_from_spec(spec), layout=layout,
                               bilingual=bilingual))

    # the fonts that will really be burned in (free-for-commercial-use, covering the text)
    return FontLibrary.load().enforce(text, proj["config"].get("target", ""), proj["config"].get("source", ""))[0]


@app.get("/api/dubtrans/projects/{pid}/style")
def dt_style(pid: str, dub: str = "", bilingual: bool = False):
    from subalign.style import PRESETS

    proj = _dt_get(pid)
    return {"spec": proj.get("style") or _default_style_spec(proj), "targets": [{"ass": "preview.ass"}], "presets": list(PRESETS)}


@app.post("/api/dubtrans/projects/{pid}/style")
def dt_style_render(pid: str, dub: str = "", bilingual: bool = False, body: Dict[str, Any] = Body(...)):
    """Preview the subtitles with ``spec``; ``save`` keeps it for the next video."""
    spec = body.get("spec") or _default_style_spec(_dt_get(pid))
    text = _dv_ass(pid, dub or None, bilingual, spec)
    if not body.get("save"):
        return {"ass": text, "target": "preview.ass"}
    with _dt_lock:
        proj = _dt_get(pid)
        proj["style"] = spec
        _dt_save(pid, proj)
    return {"ass": text, "target": "preview.ass", "written": ["字幕样式"]}


@app.get("/api/dubtrans/projects/{pid}/preview.ass")
def dt_preview_ass(pid: str, dub: str = "", bilingual: bool = False):
    from fastapi.responses import PlainTextResponse

    proj = _dt_get(pid)
    return PlainTextResponse(_dv_ass(pid, dub or None, bilingual, proj.get("style") or _default_style_spec(proj)))


@app.get("/api/fonts/free")
def free_fonts():
    """The free-for-commercial-use font library (translated subtitles use only these)."""
    from subalign.fontlib import FontLibrary

    lib = FontLibrary.load()
    out = [{"names": list(dict.fromkeys([e.family, e.display] + e.names)), "url": f"/fonts/free/{e.id}",
            "preview": f"/fonts/preview/free/{e.id}.png",
            "display": e.display, "scripts": e.scripts, "license": e.license, "verified": e.verified, "note": e.note}
           for e in lib.entries]
    return out


@app.get("/fonts/free/{fid}")
def free_font_file(fid: str):
    from subalign.fontlib import FontLibrary

    lib = FontLibrary.load()
    e = next((x for x in lib.entries if x.id == fid), None)
    if fid == "fallback":
        e = lib.default_for("zh")
    if e is None:
        raise HTTPException(404)
    p = lib.path(e)
    return FileResponse(p, media_type="font/collection" if p.suffix.lower() in (".ttc", ".otc") else "font/ttf")


def _dv_worker() -> None:
    from subalign.dubvideo import ComposeConfig, VideoSpec, compose

    while True:
        pid, vid = _dv_queue.get()
        st = _dv_state.setdefault(vid, {})
        st.update(status="running", stage="start", progress=0.0, error=None)
        try:
            proj = _dt_get(pid)
            v = next(x for x in proj.get("videos", []) if x["id"] == vid)
            dp = DUB_DIR / Path(v["dub"]).name
            if not (dp / "mix.wav").exists():
                raise RuntimeError("这个配音项目还没有合成总音频")
            c = dict(v["compose"])
            spec = VideoSpec(**{k: x for k, x in (c.pop("spec", None) or {}).items() if k in VideoSpec.__dataclass_fields__})
            cfg = ComposeConfig(**{k: x for k, x in c.items() if k in ComposeConfig.__dataclass_fields__}, spec=spec)
            style_spec = v.get("style") or _default_style_spec(proj)
            ollama.unload_all()                        # separation needs the GPU
            title = "".join(ch for ch in (proj.get("title") or "video") if ch.isalnum() or ch in "-_ ").strip()[:40] or "video"
            res = compose(Path(proj["source"]["media_path"]), dp / "mix.wav", _dv_sentences(proj, v["dub"]),
                          DT_DIR / pid / "video" / vid, cfg, stems_dir=DT_DIR / pid / "stems",
                          style=_style_from_spec(style_spec),
                          font_size=int(style_spec["font_size"]) if style_spec.get("font_size") else None,
                          source_language=proj["config"].get("source", ""),
                          base=f"{title}.{proj['config'].get('target', 'tr')}", language=proj["config"].get("target", ""),
                          progress=lambda stage, f: st.update(stage=stage, progress=round(f, 3)))
            files = {k: p.name for k, p in res["files"].items()}
            with _dt_lock:
                p2 = _dt_get(pid)
                for x in p2.get("videos", []):
                    if x["id"] == vid:
                        x.update(status="done", files=files, report=res["report"], finished=time.time())
                _dt_save(pid, p2)
            st.update(status="done", stage="done", progress=1.0)
        except Exception as e:
            log.exception("translated video %s failed", vid)
            st.update(status="error", error=f"{type(e).__name__}: {e}"[:1500])
            with _dt_lock:
                p2 = _dt_get(pid)
                for x in p2.get("videos", []):
                    if x["id"] == vid:
                        x.update(status="error", error=st["error"])
                _dt_save(pid, p2)


@app.post("/api/dubtrans/projects/{pid}/video")
def dt_video(pid: str, body: Dict[str, Any] = Body(...)):
    """Make the translated video: ``dub`` (voice-over project), ``compose`` (background /
    subtitles / output spec); the current subtitle style is used."""
    proj = _dt_get(pid)
    if not _dt_video_info(proj["source"].get("media_path")):
        raise HTTPException(400, "源文件不是视频")
    did = body.get("dub") or ""
    info = _dub_summary(did)
    if not info or did not in (proj.get("dubbing") or []):
        raise HTTPException(400, "请选择这个翻译生成的配音项目")
    if not info["mix"]:
        raise HTTPException(400, "配音还没有合成总音频（在 AI 配音页面生成完全部句子）")
    vid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    with _dt_lock:
        p2 = _dt_get(pid)
        same = [v for v in p2.get("videos", []) if v["id"] in _dv_state
                and _dv_state[v["id"]].get("status") in ("queued", "running") and v["dub"] == did
                and v.get("compose") == (body.get("compose") or {}) and v.get("style") == p2.get("style")]
        if same:
            raise HTTPException(409, "相同设置的翻译视频已经在生成中，请等它完成")
        p2.setdefault("videos", []).append({"id": vid, "created": time.time(), "dub": did, "status": "queued",
                                            "compose": body.get("compose") or {}, "style": p2.get("style")})
        _dt_save(pid, p2)
    _dv_state[vid] = {"status": "queued", "stage": "queued", "progress": 0.0, "error": None}
    _dv_queue.put((pid, vid))
    return _dt_view(pid)


@app.get("/dt/{pid}/{rel:path}")
def dt_file(pid: str, rel: str):
    base = _dt_dir(pid).resolve()
    p = (base / rel).resolve()
    if base not in p.parents or not p.is_file():
        raise HTTPException(404)
    return FileResponse(p)


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
    threading.Thread(target=_tts_worker, daemon=True).start()
    threading.Thread(target=_dv_worker, daemon=True).start()
    if not a.no_ollama and paths.OLLAMA_EXE.exists():
        threading.Thread(target=ollama.ensure_server, daemon=True).start()
    print(f"subalign 功能测试台: http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
