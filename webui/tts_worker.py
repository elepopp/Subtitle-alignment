"""IndexTTS-2.5 worker - runs inside tools/index-tts/.venv (torch 2.8, its own deps).

The main app starts it once and keeps it alive; the model stays loaded between
requests.  Protocol: one JSON request per stdin line, one JSON reply per line on
stdout prefixed with ``@@RESULT `` (the library prints its own progress lines).

    {"id": 1, "cmd": "synth", "text": "...", "spk": "a.wav", "emo": "b.wav" | null,
     "emo_alpha": 0.8, "duration_factor": 1.0, "seed": 1234, "out": "seg.wav", "lang": "ZH",
     "timbre": "voice.wav" | null}

``timbre`` splits the two stages of IndexTTS-2.5: the language model (content, rhythm,
tone) is conditioned on ``spk`` as usual, the acoustic renderer (s2mel, which largely
decides the timbre) on ``timbre`` instead - a dubbed sentence takes its delivery from
its own original line and its voice from the speaker's stable reference.
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
_timbre = {"path": None, "cache": {}}     # the timbre reference of the current request


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
        _wrap_renderer(_tts)
        print(f">> model loaded in {time.time() - t:.1f}s (bf16={bf16})", file=sys.stderr, flush=True)
    return _tts


def _timbre_condition(tts, path):
    """(prompt condition, reference mel, CAM++ style) of a reference for the acoustic
    renderer - computed the way ``IndexTTS2.infer`` does for its speaker prompt."""
    if path in _timbre["cache"]:
        return _timbre["cache"][path]
    import torch
    import torchaudio

    with torch.no_grad():
        audio, sr = tts._load_and_cut_audio(path, 15, False)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)
        inputs = tts.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
        emb = tts.get_emb(inputs["input_features"].to(tts.device), inputs["attention_mask"].to(tts.device))
        ref_mel = tts.mel_fn(audio_22k.to(emb.device).float())
        feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device), num_mel_bins=80, dither=0,
                                                 sample_frequency=16000)
        style = tts.campplus_model((feat - feat.mean(dim=0, keepdim=True)).unsqueeze(0))
        prompt = tts.s2mel.models["length_regulator"](emb, ylens=torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device),
                                                      n_quantizers=3, f0=None)[0]
    _timbre["cache"] = {path: (prompt, ref_mel, style)}    # one speaker at a time is plenty
    return prompt, ref_mel, style


def _wrap_renderer(tts):
    """While a request has a ``timbre``, the s2mel flow-matching step gets the timbre
    reference's prompt / mel / style in place of the speaker prompt's.  The caller then
    drops the first ``ref_mel`` frames of the output (its own prompt's length), so the
    output is padded back to that length."""
    import torch

    cfm = tts.s2mel.models["cfm"]
    original = cfm.inference

    def inference(cat_condition, lengths, ref_mel, style, f0, steps, **kw):
        if not _timbre["path"]:
            return original(cat_condition, lengths, ref_mel, style, f0, steps, **kw)
        n_own = ref_mel.size(-1)                            # the speaker prompt's frames
        prompt, t_mel, t_style = _timbre_condition(tts, _timbre["path"])
        cond = torch.cat([prompt.to(cat_condition.dtype), cat_condition[:, n_own:]], dim=1)
        out = original(cond, torch.LongTensor([cond.size(1)]).to(cond.device), t_mel, t_style, f0, steps, **kw)
        gen = out[:, :, t_mel.size(-1):]
        return torch.cat([gen.new_zeros(gen.size(0), gen.size(1), n_own), gen], dim=2)

    cfm.inference = inference


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
    _timbre["path"] = req.get("timbre") or None
    try:
        tts.infer(**kw)
    finally:
        _timbre["path"] = None
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
