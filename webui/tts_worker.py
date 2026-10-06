"""IndexTTS-2.5 worker - runs inside tools/index-tts/.venv (torch 2.8, its own deps).

The main app starts it once and keeps it alive; the model stays loaded between
requests.  Protocol: one JSON request per stdin line, one JSON reply per line on
stdout prefixed with ``@@RESULT `` (the library prints its own progress lines).

    {"id": 1, "cmd": "synth", "text": "...", "spk": "a.wav", "emo": "b.wav" | null,
     "emo_alpha": 0.8, "duration_factor": 1.0, "seed": 1234, "out": "seg.wav", "lang": "ZH"}
    {"id": 2, "cmd": "ping"}      {"id": 3, "cmd": "quit"}
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT / "tools" / "index-tts"
MODEL_DIR = Path(os.environ.get("SUBALIGN_TTS_MODEL", ROOT / "models" / "indextts-2.5"))
sys.path.insert(0, str(REPO))
os.chdir(REPO)                              # the library resolves some paths relative to the repo

_tts = None


def reply(obj) -> None:
    sys.stdout.write("@@RESULT " + json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load():
    global _tts
    if _tts is None:
        import torch
        from indextts.infer_v2_5 import IndexTTS2

        t = time.time()
        # Turing GPUs (RTX 20xx) have no fast bf16: fp32 there, bf16 on newer cards
        bf16 = bool(torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8)
        _tts = IndexTTS2(cfg_path=str(MODEL_DIR / "config.yaml"), model_dir=str(MODEL_DIR), use_bf16=bf16,
                         use_cuda_kernel=False, use_deepspeed=False)
        print(f">> model loaded in {time.time() - t:.1f}s (bf16={bf16})", file=sys.stderr, flush=True)
    return _tts


def synth(req):
    import torch

    tts = load()
    seed = int(req.get("seed") or 0)
    if seed:
        torch.manual_seed(seed)
    t = time.time()
    out = req["out"]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    kw = dict(spk_audio_prompt=req["spk"], text=req["text"], output_path=out, lang=req.get("lang", "ZH"),
              duration_factor=float(req.get("duration_factor", 1.0)), interval_silence=0,
              max_text_tokens_per_segment=int(req.get("max_tokens", 120)), verbose=False)
    if req.get("emo"):
        kw.update(emo_audio_prompt=req["emo"], emo_alpha=float(req.get("emo_alpha", 0.8)))
    for k in ("temperature", "top_p", "top_k"):
        if k in req:
            kw[k] = req[k]
    tts.infer(**kw)
    import soundfile as sf

    info = sf.info(out)
    return {"out": out, "seconds": round(time.time() - t, 2), "duration": round(info.duration, 3), "sr": info.samplerate}


def main():
    reply({"ready": True, "model_dir": str(MODEL_DIR)})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        rid = req.get("id")
        try:
            if req.get("cmd") == "quit":
                reply({"id": rid, "ok": True})
                break
            if req.get("cmd") == "ping":
                reply({"id": rid, "ok": True, "loaded": _tts is not None})
            elif req.get("cmd") == "load":
                load()
                reply({"id": rid, "ok": True})
            elif req.get("cmd") == "synth":
                reply({"id": rid, "ok": True, **synth(req)})
            else:
                reply({"id": rid, "ok": False, "error": f"unknown cmd {req.get('cmd')}"})
        except Exception as e:  # report and keep serving
            reply({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})


if __name__ == "__main__":
    main()
