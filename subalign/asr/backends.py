"""Concrete ASR backends.  All imports are lazy so that only the backend you use
needs to be installed.

=================  ==============================  ==================================
name               install                         notes
=================  ==============================  ==================================
faster-whisper     pip install faster-whisper      default; word timestamps, VAD
whisper            pip install openai-whisper      reference implementation
openai             (none - HTTP API)               any OpenAI-compatible
                                                   /audio/transcriptions endpoint
funasr             pip install funasr modelscope   Paraformer: excellent Chinese,
                                                   native per-character timestamps
=================  ==============================  ==================================
"""
from __future__ import annotations

import json
import os
import uuid
import urllib.request
from typing import List, Optional

from ..text.tokenize import normalize_key, tokenize
from .base import ASRBackend, Segment, Transcript, Word


def _preload_torch_cudnn() -> None:
    """ctranslate2 ships its own cudnn64_9 loader; if it is loaded first and torch's
    (older) cuDNN sub-libraries are found later, the process aborts with "Could not
    load symbol cudnnGetLibConfig".  Loading torch's cuDNN first keeps one consistent set."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.backends.cudnn.version()
    except Exception:
        pass


class FasterWhisperBackend(ASRBackend):
    name = "faster-whisper"

    def __init__(self, model: str = "large-v3", device: str = "auto", compute_type: str = "default"):
        _preload_torch_cudnn()
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install faster-whisper") from e
        self.model = WhisperModel(model, device=device, compute_type=compute_type)

    def transcribe(self, audio_path, language=None, prompt=None, vad: bool = True, beam_size: int = 5, **kw):
        segs, info = self.model.transcribe(
            str(audio_path), language=language, initial_prompt=prompt, word_timestamps=True,
            vad_filter=vad, beam_size=beam_size, condition_on_previous_text=kw.get("condition", False))
        out = []
        for s in segs:
            words = [Word(w.word, float(w.start), float(w.end), float(getattr(w, "probability", 1.0)))
                     for w in (s.words or [])]
            out.append(Segment(float(s.start), float(s.end), s.text, words))
        return Transcript(out, getattr(info, "language", language))


class WhisperBackend(ASRBackend):
    name = "whisper"

    def __init__(self, model: str = "large-v3", device: Optional[str] = None):
        try:
            import whisper  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install openai-whisper") from e
        self.model = whisper.load_model(model, device=device)

    def transcribe(self, audio_path, language=None, prompt=None, **kw):
        res = self.model.transcribe(str(audio_path), language=language, initial_prompt=prompt,
                                    word_timestamps=True, condition_on_previous_text=False)
        out = []
        for s in res.get("segments", []):
            words = [Word(w["word"], float(w["start"]), float(w["end"]), float(w.get("probability", 1.0)))
                     for w in s.get("words", [])]
            out.append(Segment(float(s["start"]), float(s["end"]), s["text"], words))
        return Transcript(out, res.get("language", language))


class OpenAIAPIBackend(ASRBackend):
    """OpenAI-compatible ``/audio/transcriptions`` (verbose_json + word timestamps)."""

    name = "openai"

    def __init__(self, model: str = "whisper-1", base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def transcribe(self, audio_path, language=None, prompt=None, **kw):
        fields = {"model": self.model, "response_format": "verbose_json"}
        if language:
            fields["language"] = language
        if prompt:
            fields["prompt"] = prompt[:800]
        boundary = uuid.uuid4().hex
        body = bytearray()
        for k, v in fields.items():
            body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
        for g in ("word", "segment"):
            body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"timestamp_granularities[]\"\r\n\r\n"
                     f"{g}\r\n").encode()
        fname = os.path.basename(str(audio_path))
        body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\n"
                 f"Content-Type: application/octet-stream\r\n\r\n").encode()
        with open(audio_path, "rb") as f:
            body += f.read()
        body += f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"{self.base_url}/audio/transcriptions", data=bytes(body), method="POST",
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": f"multipart/form-data; boundary={boundary}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = json.loads(r.read().decode("utf-8"))
        return parse_verbose_json(data, language)


def parse_verbose_json(data: dict, language: Optional[str] = None) -> Transcript:
    words = [Word(w["word"], float(w["start"]), float(w["end"])) for w in data.get("words", [])]
    segs = data.get("segments") or [{"start": 0.0, "end": float(data.get("duration", 0)), "text": data.get("text", "")}]
    out = []
    wi = 0
    for s in segs:
        seg = Segment(float(s["start"]), float(s["end"]), s.get("text", ""))
        while wi < len(words) and words[wi].start < seg.end - 1e-3:
            w = words[wi]
            if not w.text.startswith(" ") and seg.words and not _is_cjk_word(w.text):
                w.text = " " + w.text
            seg.words.append(w)
            wi += 1
        out.append(seg)
    if wi < len(words) and out:
        out[-1].words.extend(words[wi:])
    return Transcript(out, data.get("language", language))


def _is_cjk_word(s: str) -> bool:
    from ..text.tokenize import is_cjk

    return bool(s) and is_cjk(s.strip()[:1] or "a")


class FunASRBackend(ASRBackend):
    """FunASR Paraformer (best open model for Mandarin, native char timestamps)."""

    name = "funasr"

    def __init__(self, model: str = "paraformer-zh", device: Optional[str] = None):
        try:
            from funasr import AutoModel  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install funasr modelscope") from e
        kw = {"model": model, "vad_model": "fsmn-vad", "punc_model": "ct-punc"}
        if device:
            kw["device"] = device
        self.model = AutoModel(**kw)

    def transcribe(self, audio_path, language=None, prompt=None, **kw):
        res = self.model.generate(input=str(audio_path), sentence_timestamp=True)
        segs: List[Segment] = []
        for r in res:
            infos = r.get("sentence_info") or [{"text": r.get("text", ""), "timestamp": r.get("timestamp", []),
                                               "start": None, "end": None}]
            for si in infos:
                segs.append(_funasr_segment(si.get("text", ""), si.get("timestamp", []) or []))
        return Transcript([s for s in segs if s.words], language or "zh")


def _funasr_segment(text: str, stamps: list) -> Segment:
    toks = [t for t in tokenize(text) if normalize_key(t.text)]
    words: List[Word] = []
    if stamps and len(stamps) == len(toks):
        for t, (a, b) in zip(toks, stamps):
            words.append(Word(t.text + (" " if t.space_after else ""), a / 1000.0, b / 1000.0))
    elif stamps:
        a, b = stamps[0][0] / 1000.0, stamps[-1][1] / 1000.0
        step = (b - a) / max(1, len(toks))
        for k, t in enumerate(toks):
            words.append(Word(t.text, a + k * step, a + (k + 1) * step, 0.5))
    start = words[0].start if words else 0.0
    end = words[-1].end if words else 0.0
    return Segment(start, end, text, words)


BACKENDS = {
    "faster-whisper": FasterWhisperBackend,
    "whisper": WhisperBackend,
    "openai": OpenAIAPIBackend,
    "funasr": FunASRBackend,
}


def get_backend(name: str = "auto", **kw) -> ASRBackend:
    if name == "auto":
        errors = []
        for cand in ("faster-whisper", "whisper", "funasr"):
            try:
                return BACKENDS[cand](**kw)
            except ImportError as e:
                errors.append(str(e))
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIAPIBackend(**{k: v for k, v in kw.items() if k in ("model", "base_url", "api_key")})
        raise ImportError("No ASR backend available. Install one of: " + "; ".join(errors))
    if name not in BACKENDS:
        raise ValueError(f"unknown ASR backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](**kw)
