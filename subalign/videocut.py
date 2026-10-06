"""Video rough cut (视频粗剪): the speech rough cut for talking-head / screen videos,
cut so that the picture does not fall apart.

The speech rough cut (:mod:`subalign.roughcut`, unchanged) cuts the picture wherever
it cuts the sound, so every 嗯 becomes a jump cut.  Here every removal is first
judged as a *picture* edit:

1. **decide per removal** (``mode``):

   * ``cut``  - jump cut (retakes, long pauses, redundant sentences, repeats ...)
   * ``mute`` - a short filler is replaced by room tone, the picture runs on
     (only when the mouth hardly moves while saying it)
   * ``keep`` - the removal is given up

   Constraints: no picture segment shorter than ``min_shot`` and at most
   ``max_cuts`` cuts in any 10 s; when they are violated the least valuable
   cut (filler < unrecognised < repeat < pause < redundant < retake; longer is
   more valuable) is muted or given up.  Retakes and long pauses are never given up.

2. **place the picture cut**: inside a pause nobody speaks, so the removed
   window of the *picture* may slide (same length, at most ``slide``) to the
   pair of frames that look most alike - lip sync is untouched because only
   silent frames move.  The frames shown while the joint is padded with room
   tone are taken from silence as well (the speech rough cut shows the removed
   filler there).

3. **hide what is still visible** (``transition``): the jump is measured
   against the video's own frame-to-frame motion - small: hard cut, medium: a
   few-frame dissolve, large: a punch-in (alternating 100 % / ``zoom`` framing
   centred on the face, which reads as a second camera).  Screen recordings
   (static picture, no face) are always hard cut - a jump there is invisible.

Rendering decodes the source at a constant frame rate (phone videos are VFR),
maps every output frame to a whole source frame (sync within half a frame) and
encodes with the rendered audio.  The plan (``.videocut.json``) can be edited and
rendered again; an FCPXML timeline is written for Final Cut / DaVinci / Premiere.
"""
from __future__ import annotations

import bisect
import json
import logging
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import numpy as np

from . import roughcut as rc
from .roughcut import Item, RoughCutConfig

log = logging.getLogger("subalign")

# what a removal is worth (a cut that is given up costs this much)
VALUE = {"retake": 5.0, "manual": 5.0, "llm": 4.0, "repeat": 2.5, "cough": 2.0, "unrecognized": 1.2,
         "filler": 1.0, "breath": 0.8}
MUTABLE = {"filler", "unrecognized", "breath"}          # short sounds that may be silenced instead of cut
NOT_SPEECH = {"breath"}                                 # the mouth does not "speak" these
MODES = ("cut", "mute", "keep")
TRANSITIONS = ("auto", "cut", "fade", "zoom")
WHY = {"short_shot": "画面段过短", "density": "剪点过密", "manual": "手动"}


@dataclass
class VideoCutConfig:
    min_shot: float = 1.5           # shortest picture segment between two cuts (s)
    max_cuts: int = 3               # at most this many picture cuts in any 10 s
    protect: float = 3.0            # removals worth this much are never given up
    mute: bool = True               # short fillers may be silenced instead of cut
    mute_max: float = 0.45          # longest filler that may be silenced (s)
    mouth_still: float = 0.6        # mouth motion while saying it / while talking, below which it is silenced
    slide: float = 0.25             # how far a picture cut may move inside the silence (s)
    transitions: str = "auto"       # auto | cut | fade | zoom
    jump_soft: float = 3.0          # jump (x typical frame-to-frame change) below which a hard cut is invisible
    jump_hard: float = 8.0          # above it a punch-in hides the jump (between: dissolve)
    fade_frames: int = 4
    zoom: float = 1.12
    content: str = "auto"           # auto | talking | screen
    crf: int = 20
    preset: str = "veryfast"
    fcpxml: bool = True


@dataclass
class Removal:
    start: float                    # source span removed from the sound
    end: float
    fill: float = 0.0               # room tone at the joint (s)
    reasons: List[str] = field(default_factory=list)
    text: str = ""
    speech: float = 0.0             # seconds of removed speech
    value: float = 0.0
    edge: bool = False              # head / tail of the file
    mutable: bool = False
    mouth: Optional[float] = None   # mouth motion while saying it / while talking
    mode: str = "cut"
    auto_mode: str = "cut"
    why: str = ""
    locked: bool = False            # mode set by the user
    transition: str = "auto"        # requested (auto = decided here)
    applied: str = ""               # cut | fade | zoom actually used
    shift: float = 0.0              # picture cut moved by this much (s)
    jump: Optional[float] = None
    out_time: Optional[float] = None


# ------------------------------------------------------------------ media
def _ffmpeg() -> str:
    try:
        from .studio import _ffmpeg as f

        return f()
    except Exception:
        return shutil.which("ffmpeg") or "ffmpeg"


def _ffprobe() -> str:
    ff = Path(_ffmpeg())
    cand = ff.with_name(ff.name.replace("ffmpeg", "ffprobe"))
    return str(cand) if cand.exists() else (shutil.which("ffprobe") or "ffprobe")


def video_info(path: Path) -> Dict[str, Any]:
    """Display size (rotation applied), frame rate (as a fraction) and duration."""
    out = subprocess.run([_ffprobe(), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                         capture_output=True, check=True, text=True, encoding="utf-8").stdout
    d = json.loads(out)
    vs = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"
               and not s.get("disposition", {}).get("attached_pic")), None)
    if vs is None:
        raise ValueError("视频粗剪需要视频文件（没有找到视频轨）")
    w, h = int(vs["width"]), int(vs["height"])
    rot = 0
    for sd in vs.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    rot = int(float((vs.get("tags") or {}).get("rotate", rot)))
    if abs(rot) % 180 == 90:
        w, h = h, w
    fr = None
    for k in ("avg_frame_rate", "r_frame_rate"):
        try:
            f = Fraction(vs.get(k) or "0/1")
            if 5 <= f <= 120:
                fr = f
                break
        except (ValueError, ZeroDivisionError):
            continue
    fr = (fr or Fraction(30)).limit_denominator(1001)
    dur = float(d.get("format", {}).get("duration") or vs.get("duration") or 0)
    return {"width": w, "height": h, "fps": fr, "duration": dur}


def _small_size(w: int, h: int, n: int = 160) -> Tuple[int, int]:
    if w >= h:
        return n, max(2, int(round(n * h / w / 2)) * 2)
    return max(2, int(round(n * w / h / 2)) * 2), n


# ------------------------------------------------------------------ video analysis
class FrameStore:
    """Small grey frames around the removals, frame-to-frame motion of the whole
    video and face boxes - everything the picture decisions need."""

    def __init__(self, fps: float, size: Tuple[int, int]):
        self.fps = fps
        self.size = size
        self.frames: Dict[int, np.ndarray] = {}
        self.motion = np.zeros(0, np.float32)
        self.faces: Dict[int, Optional[Tuple[float, float, float, float]]] = {}
        self.content = "talking"

    @property
    def typical(self) -> float:
        m = self.motion[self.motion > 0]
        return max(float(np.median(m)) if len(m) else 0.0, 0.004)

    def frame(self, i: int) -> Optional[np.ndarray]:
        if i in self.frames:
            return self.frames[i]
        if not self.frames:
            return None
        keys = sorted(self.frames)
        k = keys[min(len(keys) - 1, bisect.bisect_left(keys, i))]
        return self.frames[k] if abs(k - i) <= 1 else None

    def jump(self, a: int, b: int) -> Optional[float]:
        """How different two frames are, in units of the typical frame-to-frame change."""
        fa, fb = self.frame(a), self.frame(b)
        if fa is None or fb is None:
            return None
        d = float(np.mean(np.abs(fa.astype(np.float32) - fb.astype(np.float32)))) / 255.0
        return round(d / self.typical, 2)

    def face_near(self, t: float, span: float = 1.0) -> Optional[Tuple[float, float, float, float]]:
        lo, hi = int((t - span) * self.fps), int((t + span) * self.fps)
        seen = [b for i, b in self.faces.items() if lo <= i <= hi]
        boxes = [b for b in seen if b]
        # a real face is found in most frames, at about the same place; stray hits are texture
        if len(boxes) < max(2, 0.4 * len(seen)):
            return None
        cx = np.array([b[0] + b[2] / 2 for b in boxes])
        if float(np.std(cx)) > 0.1:
            return None
        return tuple(float(np.median([b[k] for b in boxes])) for k in range(4))  # type: ignore[return-value]

    def mouth_motion(self, t0: float, t1: float, face) -> Optional[float]:
        """Mean frame-to-frame change in the mouth region (lower part of the face box)."""
        if face is None:
            return None
        W, H = self.size
        x, y, w, h = face
        xa, xb = int((x + 0.2 * w) * W), int((x + 0.8 * w) * W)
        ya, yb = int((y + 0.62 * h) * H), int(min(1.0, y + 1.0 * h) * H)
        if xb - xa < 2 or yb - ya < 2:
            return None
        vals = []
        for i in range(int(t0 * self.fps), int(t1 * self.fps) + 1):
            a, b = self.frames.get(i - 1), self.frames.get(i)
            if a is not None and b is not None:
                vals.append(float(np.mean(np.abs(a[ya:yb, xa:xb].astype(np.float32) - b[ya:yb, xa:xb]))))
        return float(np.mean(vals)) if vals else None


def _detect_faces(store: FrameStore, idxs: Sequence[int]) -> None:
    try:
        import cv2
    except ImportError:  # pragma: no cover - optional
        log.info("opencv not installed: no face detection (punch-ins centred, fillers muted by length only)")
        return
    casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    W, H = store.size
    for i in idxs:
        f = store.frames.get(i)
        if f is None:
            continue
        found = casc.detectMultiScale(f, scaleFactor=1.1, minNeighbors=5, minSize=(max(12, W // 12),) * 2)
        if len(found):
            x, y, w, h = max(found, key=lambda b: b[2] * b[3])
            store.faces[i] = (x / W, y / H, w / W, h / H)
        else:
            store.faces[i] = None


def analyze_video(path: Path, info: Dict[str, Any], windows: Sequence[Tuple[float, float]],
                  sample_every: float = 1.0) -> FrameStore:
    """Decode small grey frames at a constant rate: motion for every frame, frames
    kept inside ``windows`` (source seconds) and once per ``sample_every``."""
    fps = float(info["fps"])
    W, H = _small_size(info["width"], info["height"])
    store = FrameStore(fps, (W, H))
    spans = sorted((max(0, int(a * fps)), int(b * fps) + 1) for a, b in windows)
    every = max(1, int(round(sample_every * fps)))
    cmd = [_ffmpeg(), "-nostdin", "-v", "error", "-i", str(path), "-an", "-vf",
           f"fps={info['fps']},scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = W * H
    motion: List[float] = []
    prev = None
    i, k = 0, 0
    assert p.stdout is not None
    while True:
        buf = p.stdout.read(n)
        if len(buf) < n:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W)
        motion.append(float(np.mean(np.abs(f.astype(np.int16) - prev))) / 255.0 if prev is not None else 0.0)
        prev = f.astype(np.int16)
        while k < len(spans) and spans[k][1] <= i:
            k += 1
        if (k < len(spans) and spans[k][0] <= i < spans[k][1]) or i % every == 0:
            store.frames[i] = f
        i += 1
    p.wait()
    store.motion = np.asarray(motion, np.float32)
    samples = [j for j in store.frames if j % every == 0]
    in_win = [j for j in store.frames if j % every and j % 3 == 0]
    _detect_faces(store, samples + in_win)
    seen = [store.faces.get(j) for j in samples]
    face_share = sum(1 for b in seen if b) / max(1, len(seen))
    store.content = "talking" if face_share >= 0.3 or float(np.median(store.motion) if len(store.motion) else 0) > 0.002 \
        else "screen"
    log.info("video: %d frames, typical motion %.4f, faces in %.0f%% of samples -> %s",
             i, store.typical, face_share * 100, store.content)
    return store


# ------------------------------------------------------------------ deciding
def speech_spans(items: Sequence[Item]) -> List[Tuple[float, float]]:
    """Where the mouth is busy: every recognised unit (kept or removed) and every
    unrecognised voicing; breaths do not count."""
    out = [(it.start, it.end) for it in items if (it.key or it.island) and (it.cut or it.auto) not in NOT_SPEECH]
    return sorted(out)


def removals_from_keeps(keeps: Sequence[Tuple[float, float]], fills: Sequence[float], items: Sequence[Item],
                        duration: float) -> List[Removal]:
    out: List[Removal] = []
    bounds = []
    if keeps and keeps[0][0] > 1e-3:
        bounds.append((0.0, keeps[0][0], 0.0, True))
    for k in range(len(keeps) - 1):
        bounds.append((keeps[k][1], keeps[k + 1][0], fills[k], False))
    if keeps and duration - keeps[-1][1] > 1e-3:
        bounds.append((keeps[-1][1], duration, 0.0, True))
    for a, b, fill, edge in bounds:
        drops = [it for it in items if it.cut and a - 0.02 <= (it.start + it.end) / 2 <= b + 0.02]
        reasons = sorted({it.cut for it in drops}, key=lambda r: -VALUE.get(r, 1.0)) or ["pause"]
        speech = sum(it.end - it.start for it in drops if it.cut not in NOT_SPEECH)
        r = Removal(start=round(a, 4), end=round(b, 4), fill=round(fill, 4), reasons=reasons, edge=edge,
                    text="".join(it.text for it in sorted(drops, key=lambda x: x.start) if not it.island)
                    or ("（停顿）" if not drops else "（未识别发声）"), speech=round(speech, 3))
        base = max(VALUE.get(x, 1.0) for x in reasons) if drops else 1.0 + 1.5 * max(0.0, (b - a) - 0.4)
        r.value = round(base + 0.8 * (b - a), 3)
        r.mutable = bool(drops) and set(reasons) <= MUTABLE
        out.append(r)
    return out


def assess_mutes(rems: Sequence[Removal], items: Sequence[Item], store: Optional[FrameStore], cfg: VideoCutConfig) -> None:
    """A filler may be silenced instead of cut when it is short and the mouth hardly moves."""
    talk = [it for it in items if it.key and not it.cut and not it.island]
    for r in rems:
        if r.edge or not r.mutable or not cfg.mute:
            r.mutable = False
            continue
        if r.speech > cfg.mute_max:
            r.mutable = False
            continue
        if store is None or store.content == "screen":
            continue
        face = store.face_near((r.start + r.end) / 2)
        drops = [it for it in items if it.cut and r.start - 0.02 <= (it.start + it.end) / 2 <= r.end + 0.02]
        if face is None:
            r.mutable = r.speech <= 0.3                     # cannot see the mouth: only very short ones
            continue
        m_drop = [m for it in drops if (m := store.mouth_motion(it.start, it.end, face)) is not None]
        near = [it for it in talk if r.start - 1.5 <= it.start <= r.end + 1.5]
        m_talk = [m for it in near if (m := store.mouth_motion(it.start, it.end, face)) is not None]
        if m_drop and m_talk and np.median(m_talk) > 0:
            r.mouth = round(float(np.mean(m_drop)) / float(np.median(m_talk)), 2)
            r.mutable = r.mouth <= cfg.mouth_still
        else:
            r.mutable = r.speech <= 0.3


def choose_modes(rems: List[Removal], cfg: VideoCutConfig, content: str = "talking",
                 duration: Optional[float] = None) -> None:
    """Give up / silence the least valuable cuts until no picture segment is shorter
    than ``min_shot`` and no 10 s window holds more than ``max_cuts`` cuts."""
    for r in rems:
        if not r.locked:
            r.mode, r.why = "cut", ""
    inner = sorted((r for r in rems if not r.edge), key=lambda r: r.start)
    if content == "screen":                              # jump cuts are invisible on a static picture
        for r in rems:
            r.auto_mode = r.mode
        return
    first = next((r.end for r in rems if r.edge and r.start <= 1e-3), 0.0)
    last = next((r.start for r in rems if r.edge and r.start > 1e-3),
                duration if duration is not None else (inner[-1].end + 1e9 if inner else 0.0))

    def demotable(r: Optional[Removal]) -> bool:
        return r is not None and not r.locked and r.mode == "cut" and r.value < cfg.protect

    def demote(r: Removal, why: str) -> None:
        r.mode = "mute" if r.mutable else "keep"
        r.why = why

    while True:                                          # the shortest picture segment first
        cuts = [r for r in inner if r.mode == "cut"]
        if not cuts:
            break
        shots, prev, left = [], first, None
        for c in cuts:
            shots.append((c.start - prev, left, c))
            prev, left = c.end, c
        shots.append((last - prev, left, None))
        bad = [(L, min((x for x in (a, b) if demotable(x)), key=lambda x: x.value))
               for L, a, b in shots if L < cfg.min_shot and (demotable(a) or demotable(b))]
        if not bad:
            break
        demote(min(bad, key=lambda x: x[0])[1], "short_shot")
    while True:                                          # density
        cuts = [r for r in inner if r.mode == "cut"]
        over = None
        for k, r in enumerate(cuts):
            win = [x for x in cuts[k:] if x.start < r.start + 10.0]
            cand = [x for x in win if demotable(x)]
            if len(win) > cfg.max_cuts and cand:
                over = min(cand, key=lambda x: x.value)
                break
        if over is None:
            break
        demote(over, "density")
    for r in rems:
        r.auto_mode = r.mode


def final_keeps(rems: Sequence[Removal], duration: float) -> Tuple[List[Tuple[float, float]], List[float], List[Removal]]:
    """Keep segments / room-tone fills of the sound, from the removals that are cut."""
    cuts = sorted((r for r in rems if r.mode == "cut" or r.edge), key=lambda r: r.start)
    keeps, fills = [], []
    t = 0.0
    for r in cuts:
        if r.start - t > 1e-3:
            keeps.append((t, r.start))
            fills.append(r.fill)
        t = r.end
    if duration - t > 1e-3:
        keeps.append((t, duration))
        fills.append(0.0)
    if fills:
        fills[-1] = 0.0
    # the joints between keeps (a head removal is not a joint)
    inner = [r for r in cuts if not r.edge]
    return keeps, fills, inner


def mute_audio(y: np.ndarray, sr: int, spans: Sequence[Tuple[float, float]], room: np.ndarray) -> np.ndarray:
    """Replace ``spans`` with the recording's room tone (15 ms crossfades)."""
    y = y.copy()
    for a, b in spans:
        i0, i1 = max(0, int(a * sr)), min(y.shape[1], int(b * sr))
        n = i1 - i0
        if n <= 0:
            continue
        tone = rc._tone(room, n)
        f = min(int(0.015 * sr), n // 3)
        w = np.ones(n, np.float32)
        if f:
            w[:f] = np.linspace(0, 1, f)
            w[-f:] = np.linspace(1, 0, f)
        y[:, i0:i1] = y[:, i0:i1] * (1 - w) + tone * w
    return y


# ------------------------------------------------------------------ placing the picture cuts
@dataclass
class Shot:
    """One picture segment of the output: output frames [o0, o1) show source frame n + off."""
    o0: int
    o1: int
    off: int
    zoom: float = 1.0
    center: Tuple[float, float] = (0.5, 0.42)
    fade_in: int = 0                # dissolve frames centred on o0


def _silence_after(t: float, spans: Sequence[Tuple[float, float]], cap: float = 2.0) -> float:
    gaps = [max(0.0, a - t) for a, b in spans if b > t + 0.01]
    return min([cap] + gaps)


def _silence_before(t: float, spans: Sequence[Tuple[float, float]], cap: float = 2.0) -> float:
    gaps = [max(0.0, t - b) for a, b in spans if a < t - 0.01]
    return min([cap] + gaps)


def place_shots(keeps: Sequence[Tuple[float, float]], joints: Sequence[Removal], r: "rc.Rendered", items: Sequence[Item],
                store: Optional[FrameStore], fps: float, cfg: VideoCutConfig, content: str) -> List[Shot]:
    """Output picture segments: the cut instants slid inside the silence to the most
    alike frames, the transition chosen per cut, punch-ins alternating."""
    spans = speech_spans(items)
    total = int(round(r.audio.shape[1] / r.sr * fps))
    offs = [int(round((keeps[j][0] - r.seg_out[j]) * fps)) for j in range(len(keeps))]
    cut_frames = [0]
    for j in range(1, len(keeps)):
        rem = joints[j - 1]
        B = r.seg_bounds[j]
        e, s = keeps[j - 1][1], keeps[j][0]
        ext = (keeps[j - 1][0] + (B - r.seg_out[j - 1])) - e     # source shown past the keep end at B
        hi = _silence_after(e, spans) - ext
        lo = -_silence_before(s, spans) - (B - r.seg_out[j])
        hi, lo = min(hi, cfg.slide), max(lo, -cfg.slide)
        # stay inside the two segments
        lo = max(lo, r.seg_out[j - 1] + 2 / fps - B)
        hi = min(hi, r.seg_bounds[j + 1] - 2 / fps - B)
        if lo > hi:
            cands = [min(max(lo, -cfg.slide), cfg.slide)]
        else:
            n0, n1 = int(np.ceil((B + lo) * fps)), int(np.floor((B + hi) * fps))
            cands = [n / fps - B for n in range(n0, n1 + 1)] or [0.0]
        best = None
        for d in cands:
            n = int(round((B + d) * fps))
            jmp = store.jump(n - 1 + offs[j - 1], n + offs[j]) if store is not None else None
            score = (jmp if jmp is not None else 0.0) + 2.0 * abs(d)
            if best is None or score < best[0]:
                best = (score, n, jmp, d)
        _, n, jmp, d = best
        n = max(cut_frames[-1] + 1, min(total - 1, n))
        cut_frames.append(n)
        rem.shift, rem.jump, rem.out_time = round(n / fps - B, 3), jmp, round(n / fps, 3)
    cut_frames.append(total)

    shots: List[Shot] = []
    zoomed = False
    for j in range(len(keeps)):
        o0, o1 = cut_frames[j], cut_frames[j + 1]
        sh = Shot(o0, o1, offs[j])
        if j > 0:
            rem = joints[j - 1]
            want = rem.transition if rem.transition in ("cut", "fade", "zoom") else cfg.transitions
            if want not in ("cut", "fade", "zoom"):              # auto
                jm = rem.jump if rem.jump is not None else 0.0
                if content == "screen" or jm < cfg.jump_soft:
                    want = "cut"
                elif jm < cfg.jump_hard or content != "talking":
                    want = "fade"
                else:
                    want = "zoom"
            if want == "fade":
                # a dissolve shows a few frames past the cut on both sides: room for them?
                room = (rem.end - rem.start) * fps
                if room < cfg.fade_frames + 1 or o1 - o0 < cfg.fade_frames or o0 - shots[-1].o0 < cfg.fade_frames:
                    want = "cut"
                else:
                    sh.fade_in = cfg.fade_frames
            if want == "zoom":
                zoomed = not zoomed
            rem.applied = want
        sh.zoom = cfg.zoom if zoomed else 1.0
        if zoomed and store is not None:
            face = store.face_near((o0 + offs[j]) / fps + 0.5)
            if face:
                sh.center = (face[0] + face[2] / 2, face[1] + face[3] * 0.55)
        shots.append(sh)
    return shots


# ------------------------------------------------------------------ rendering
def _transform(img: np.ndarray, zoom: float, center: Tuple[float, float]) -> np.ndarray:
    if zoom <= 1.001:
        return img
    import cv2

    H, W = img.shape[:2]
    cw, ch = W / zoom, H / zoom
    cx = min(max(center[0] * W, cw / 2), W - cw / 2)
    cy = min(max(center[1] * H, ch / 2), H - ch / 2)
    x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
    crop = img[y0:y0 + int(round(ch)), x0:x0 + int(round(cw))]
    return cv2.resize(crop, (W, H), interpolation=cv2.INTER_LINEAR)


def frame_plan(shots: Sequence[Shot]) -> List[List[Tuple[int, float, int]]]:
    """For every output frame: [(source frame, weight, shot index)].  A dissolve of F
    frames is centred on the cut: both shots run on for F/2 frames into it."""
    out: List[List[Tuple[int, float, int]]] = [[(n + sh.off, 1.0, k)] for k, sh in enumerate(shots)
                                               for n in range(sh.o0, sh.o1)]
    for k, sh in enumerate(shots):
        if k == 0 or not sh.fade_in:
            continue
        F, a0 = sh.fade_in, sh.o0 - sh.fade_in // 2
        for q in range(F):
            n = a0 + q
            if 0 <= n < len(out):
                w = (q + 0.5) / F
                out[n] = [(n + shots[k - 1].off, 1 - w, k - 1), (n + sh.off, w, k)]
    return out


def render_video(src: Path, audio_wav: Path, dst: Path, shots: Sequence[Shot], info: Dict[str, Any],
                 cfg: VideoCutConfig) -> Path:
    """Decode at a constant frame rate, emit every output frame from whole source
    frames (punch-ins / dissolves applied), encode with the edited sound."""
    W, H = info["width"], info["height"]
    fps = info["fps"]
    plan = frame_plan(shots)
    dec = subprocess.Popen([_ffmpeg(), "-nostdin", "-v", "error", "-i", str(src), "-an", "-vf", f"fps={fps}",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen([_ffmpeg(), "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-i", str(audio_wav),
                            "-map", "0:v:0", "-map", "1:a:0", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                            "-c:v", "libx264", "-preset", cfg.preset, "-crf", str(cfg.crf), "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(dst)],
                           stdin=subprocess.PIPE)
    assert dec.stdout is not None and enc.stdin is not None
    size = W * H * 3
    # only frames some output frame shows are kept (a removed retake may be minutes long)
    top = max((max(i for i, _, _ in p) for p in plan), default=0)
    wanted = np.zeros(top + 2, bool)
    for p in plan:
        for i, _, _ in p:
            wanted[max(0, i)] = True
    cache: Dict[int, np.ndarray] = {}
    read = -1
    eof = False
    last: Optional[np.ndarray] = None
    try:
        for n, parts in enumerate(plan):
            need = max(i for i, _, _ in parts)
            while read < need and not eof:
                buf = dec.stdout.read(size)
                if len(buf) < size:
                    eof = True
                    break
                read += 1
                if wanted[read]:
                    cache[read] = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            imgs = []
            for i, w, k in parts:
                img = cache.get(max(0, i))
                if img is None:                          # past the end of the source: hold
                    img = last if last is not None else np.zeros((H, W, 3), np.uint8)
                imgs.append((_transform(img, shots[k].zoom, shots[k].center), w))
            if len(imgs) == 1:
                out = imgs[0][0]
            else:
                out = _as_u8(sum(im.astype(np.float32) * w for im, w in imgs) + 0.5)
            last = out
            enc.stdin.write(out.tobytes())
            if n + 1 < len(plan):                          # frames no later output frame needs
                floor = min(i for i, _, _ in plan[n + 1])
                for i in [i for i in cache if i < floor]:
                    del cache[i]
    finally:
        enc.stdin.close()
        dec.stdout.close()
        dec.kill()
        enc.wait()
    if enc.returncode:
        raise RuntimeError(f"ffmpeg encode failed ({enc.returncode})")
    return dst


def _as_u8(a: np.ndarray) -> np.ndarray:
    return a if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ FCPXML
def _rt(frames: int, fps: Fraction) -> str:
    v = Fraction(frames) / fps
    return f"{v.numerator}/{v.denominator}s" if v.denominator != 1 else f"{v.numerator}s"


def fcpxml(src: Path, audio: Path, shots: Sequence[Shot], joints: Sequence[Removal], info: Dict[str, Any],
           name: str) -> str:
    """Timeline for Final Cut Pro / DaVinci Resolve / Premiere: the picture as clips of
    the source (video only, punch-ins as transforms, dissolves as markers) and the
    edited sound as one clip under it."""
    from xml.sax.saxutils import quoteattr

    fps = info["fps"]
    W, H = info["width"], info["height"]
    total = shots[-1].o1 if shots else 0
    src_frames = int(round(info["duration"] * float(fps))) or total
    url = lambda p: "file:///" + quote(str(Path(p).resolve()).replace("\\", "/").lstrip("/"))  # noqa: E731
    fd = 1 / fps
    rows = ['<?xml version="1.0" encoding="UTF-8"?>', "<!DOCTYPE fcpxml>", '<fcpxml version="1.9">', "<resources>",
            f'<format id="r1" name="FFVideoFormat{H}p" frameDuration="{fd.numerator}/{fd.denominator}s" '
            f'width="{W}" height="{H}"/>',
            f'<asset id="r2" name={quoteattr(src.stem)} start="0s" duration="{_rt(src_frames, fps)}" hasVideo="1" '
            f'hasAudio="1" format="r1"><media-rep kind="original-media" src={quoteattr(url(src))}/></asset>',
            f'<asset id="r3" name={quoteattr(audio.stem)} start="0s" duration="{_rt(total, fps)}" hasAudio="1" '
            f'audioSources="1"><media-rep kind="original-media" src={quoteattr(url(audio))}/></asset>',
            "</resources>", f"<library><event name={quoteattr(name)}><project name={quoteattr(name)}>",
            f'<sequence format="r1" duration="{_rt(total, fps)}" tcStart="0s" tcFormat="NDF"><spine>']
    for k, sh in enumerate(shots):
        start = max(0, sh.o0 + sh.off)
        attrs = (f'ref="r2" name={quoteattr(src.stem)} offset="{_rt(sh.o0, fps)}" start="{_rt(start, fps)}" '
                 f'duration="{_rt(sh.o1 - sh.o0, fps)}" srcEnable="video"')
        inner = []
        if k == 0:
            inner.append(f'<asset-clip ref="r3" lane="-1" offset="{_rt(start, fps)}" start="0s" '
                         f'duration="{_rt(total, fps)}" name="edited audio"/>')
        if sh.zoom > 1.001:
            # position: offset of the frame centre in % of the frame height
            px = (0.5 - sh.center[0]) * (sh.zoom - 1) * W / H * 100
            py = (sh.center[1] - 0.5) * (sh.zoom - 1) * 100
            inner.append(f'<adjust-transform scale="{sh.zoom:.3f} {sh.zoom:.3f}" position="{px:.2f} {py:.2f}"/>')
        if k > 0 and sh.fade_in:
            inner.append(f'<marker start="{_rt(start, fps)}" duration="{_rt(1, fps)}" value="叠化 {sh.fade_in} 帧"/>')
        rows.append(f"<asset-clip {attrs}>" + "".join(inner) + "</asset-clip>")
    rows.append("</spine></sequence></project></event></library></fcpxml>")
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------ report
def _ts(t: float) -> str:
    m = int(t // 60)
    return f"{m:02d}:{t - 60 * m:05.2f}"


def report_md(stats: Dict[str, Any], rems: Sequence[Removal]) -> str:
    rows = ["# 视频粗剪报告\n",
            f"原时长 {stats['duration']:.1f}s → 剪后 {stats['output_duration']:.1f}s（减少 "
            f"{stats['duration'] - stats['output_duration']:.1f}s）· 画面类型：{'录屏 / 静态画面' if stats['content'] == 'screen' else '真人出镜'}\n",
            f"- 画面剪切 {stats['cuts']} 处（硬切 {stats['transitions'].get('cut', 0)}、叠化 {stats['transitions'].get('fade', 0)}、"
            f"推近 / 拉远 {stats['transitions'].get('zoom', 0)}）",
            f"- 只静音、画面不剪 {stats['muted']} 处",
            f"- 放弃（保留原样）{stats['kept']} 处",
            f"- 最短画面段 {stats['shortest_shot']:.2f}s · 平均每分钟 {stats['cuts_per_min']:.1f} 刀\n", "## 明细\n"]
    names = {"cut": "剪", "mute": "静音", "keep": "保留"}
    tnames = {"cut": "硬切", "fade": "叠化", "zoom": "推近/拉远", "": ""}
    for r in rems:
        if r.edge:
            continue
        why = f"（{WHY.get(r.why, r.why)}）" if r.why else ""
        extra = ""
        if r.mode == "cut":
            extra = f" · {tnames.get(r.applied, r.applied)}" + (f" · 画面差异 {r.jump:g}" if r.jump is not None else "") + \
                    (f" · 剪点移动 {r.shift * 1000:+.0f}ms" if abs(r.shift) >= 0.02 else "")
        reasons = "、".join(rc.REASONS.get(x, x) for x in r.reasons)
        rows.append(f"- [{_ts(r.start)}] {names[r.mode]}{why} · {reasons}：{r.text}{extra}")
    return "\n".join(rows) + "\n"


# ------------------------------------------------------------------ entry point
def _items_for(media: Path, out_dir: Path, rcfg: RoughCutConfig, y: np.ndarray, sr: int, env, env_hop,
               script: Optional[str], language: Optional[str], asr_backend: str, asr_model: Optional[str],
               device: Optional[str], ctc: str, llm_client) -> Tuple[List[Item], Optional[str]]:
    """Recognition + detection exactly as the speech rough cut does it."""
    from .audio.features import analyze
    from .audio.io import load_audio

    if rcfg.acoustic_only:
        feats16 = analyze(load_audio(media, 16000))
        items = rc.voiced_units(feats16)
        items += rc.breaths_and_coughs(items, feats16, rcfg)
        for it in items:
            it.cut = it.auto
        return items, language
    doc = rc.transcribe_for_cut(media, script, language, asr_backend, asr_model, device, ctc, out_dir / "work")
    language = language or doc.language
    items = rc.items_from_document(doc)
    rc.acoustic_ends(items, env, env_hop)
    feats16 = analyze(load_audio(media, 16000))
    items = rc.detect(items, rcfg, feats16, llm_client)
    extra = rc.breaths_and_coughs(items, feats16, rcfg)
    for it in extra:
        it.cut = it.auto
    return items + extra, language


def video_cut(media: Path, out_dir: Path, rcfg: RoughCutConfig, vcfg: VideoCutConfig, script: Optional[str] = None,
              plan: Optional[Dict[str, Any]] = None, language: Optional[str] = None, asr_backend: str = "auto",
              asr_model: Optional[str] = None, device: Optional[str] = None, ctc: str = "auto",
              llm_client=None) -> Dict[str, Any]:
    from .audio.io import load_audio, save_audio

    media, out_dir = Path(media), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = media.stem
    info = video_info(media)
    ainfo = rc.probe(media)
    sr = ainfo["sr"]
    y = load_audio(media, sr, mono=ainfo["channels"] == 1)
    if y.ndim == 1:
        y = y[None]
    duration = y.shape[1] / sr
    env, env_hop = rc._rms_env(y, sr)

    overrides: Dict[int, Dict[str, Any]] = {}
    if plan is not None:
        items = [Item(**{k: v for k, v in d.items() if k in Item.__dataclass_fields__}) for d in plan["items"]]
        language = language or plan.get("language")
        s = plan.get("settings") or {}
        for k in ("max_pause", "keep_pause", "pauses", "min_gap", "crossfade_ms", "level", "breaths",
                  "breath_reduce_db", "acoustic_only"):
            if k in s and getattr(rcfg, k) == getattr(RoughCutConfig(), k):
                setattr(rcfg, k, s[k])
        for k, v in (s.get("video") or {}).items():
            if k in VideoCutConfig.__dataclass_fields__ and getattr(vcfg, k) == getattr(VideoCutConfig(), k):
                setattr(vcfg, k, v)
        overrides = {i: d for i, d in enumerate(plan.get("removals") or [])}
    else:
        items, language = _items_for(media, out_dir, rcfg, y, sr, env, env_hop, script, language, asr_backend,
                                     asr_model, device, ctc, llm_client)

    soft = {id(it) for it in items if it.cut == "breath" and rcfg.breaths == "reduce"}
    if soft:
        y = rc.attenuate(y, sr, [(it.start, it.end) for it in items if id(it) in soft], rcfg.breath_reduce_db)
    plan_items = [it for it in items if id(it) not in soft]
    keeps0, fills0, pause_edits = rc.plan_cuts(plan_items, duration, rcfg, env, env_hop)
    rems = removals_from_keeps(keeps0, fills0, plan_items, duration)

    # user decisions from an edited plan (matched by position)
    for i, r in enumerate(rems):
        d = overrides.get(i)
        if d is None or abs(d.get("start", -9) - r.start) > 0.08 or abs(d.get("end", -9) - r.end) > 0.08:
            d = next((x for x in overrides.values() if abs(x.get("start", -9) - r.start) <= 0.08
                      and abs(x.get("end", -9) - r.end) <= 0.08), None)
        if d:
            if d.get("locked") and d.get("mode") in MODES:
                r.mode, r.locked, r.why = d["mode"], True, "manual"
            if d.get("transition") in TRANSITIONS:
                r.transition = d["transition"]

    windows = [(r.start - vcfg.slide - 1.6, r.end + vcfg.slide + 1.6) for r in rems if not r.edge]
    store = analyze_video(media, info, windows)
    content = vcfg.content if vcfg.content in ("talking", "screen") else store.content
    assess_mutes(rems, plan_items, store, vcfg)
    choose_modes(rems, vcfg, content, duration)

    room = rc._room_tone(y, sr, env, env_hop, items)
    muted = [r for r in rems if r.mode == "mute" and not r.edge]
    mspans = [(it.start - 0.01, it.end + 0.01) for r in muted for it in plan_items
              if it.cut and r.start - 0.02 <= (it.start + it.end) / 2 <= r.end + 0.02]
    y2 = mute_audio(y, sr, mspans, room) if mspans else y
    keeps, fills, joints = final_keeps(rems, duration)
    r = rc.render(y2, sr, keeps, fills, room, env, env_hop, rcfg)
    fps = float(info["fps"])
    shots = place_shots(keeps, joints, r, plan_items, store, fps, vcfg, content)

    audio_out = save_audio(out_dir / f"{base}.vcut.wav", r.audio if r.audio.shape[0] > 1 else r.audio[0], sr)
    video_out = render_video(media, audio_out, out_dir / f"{base}.vcut.mp4", shots, info, vcfg)
    files = [video_out, audio_out]

    # transcript of the result: given-up removals are heard again, silenced ones are not
    kept_back = {id(it) for rr in rems if rr.mode == "keep" and not rr.edge for it in plan_items
                 if it.cut and rr.start - 0.02 <= (it.start + it.end) / 2 <= rr.end + 0.02}
    doc_items = [Item(**{**asdict(it), "cut": None}) if id(it) in kept_back else it for it in items]
    edited = rc.edited_document(doc_items, r, language=language)

    out_dur = r.audio.shape[1] / sr
    applied: Dict[str, int] = {}
    for x in joints:
        applied[x.applied or "cut"] = applied.get(x.applied or "cut", 0) + 1
    inner = [x for x in rems if not x.edge]
    stats = {"duration": round(duration, 3), "output_duration": round(out_dur, 3),
             "removed_ratio": round(1 - out_dur / max(duration, 1e-9), 4), "content": content,
             "cuts": len(joints), "muted": sum(1 for x in inner if x.mode == "mute"),
             "kept": sum(1 for x in inner if x.mode == "keep"), "transitions": applied,
             "shortest_shot": round(min(((s.o1 - s.o0) / fps for s in shots), default=0.0), 2),
             "cuts_per_min": round(len(joints) / max(out_dur / 60, 1e-9), 2),
             "fps": str(info["fps"]), "size": [info["width"], info["height"]]}
    plan_out = {"format": "subalign-videocut", "version": 1, "source": media.name, "language": language,
                "settings": {"level": rcfg.level, "max_pause": rcfg.resolved()["max_pause"],
                             "keep_pause": rcfg.resolved()["keep_pause"], "pauses": rcfg.pauses, "breaths": rcfg.breaths,
                             "breath_reduce_db": rcfg.breath_reduce_db, "acoustic_only": rcfg.acoustic_only,
                             "min_gap": rcfg.min_gap, "crossfade_ms": rcfg.crossfade_ms, "video": asdict(vcfg)},
                "stats": stats, "pause_edits": pause_edits, "removals": [asdict(x) for x in rems],
                "items": [asdict(it) for it in items]}
    pp = out_dir / f"{base}.videocut.json"
    pp.write_text(json.dumps(plan_out, ensure_ascii=False, indent=1), encoding="utf-8")
    rp = out_dir / f"{base}.videocut.md"
    rp.write_text(report_md(stats, rems), encoding="utf-8")
    files += [pp, rp]
    if vcfg.fcpxml:
        xp = out_dir / f"{base}.vcut.fcpxml"
        xp.write_text(fcpxml(media, audio_out, shots, joints, info, f"{base} 视频粗剪"), encoding="utf-8")
        files.append(xp)
    return {"files": files, "stats": stats, "document": edited, "base": base, "removals": rems}
