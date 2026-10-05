"""Download every model subalign can use into ./models.

    python webui/download_models.py              # everything
    python webui/download_models.py whisper ctc  # only some groups

Groups: whisper, openai-whisper, ctc, demucs, uvr, funasr, rnnoise, campplus, llm
"""
from __future__ import annotations

import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from webui import paths  # noqa: E402  (sets the cache env vars)

HF_REPOS = {
    "whisper": ["Systran/faster-whisper-large-v3", "mobiuslabsgmbh/faster-whisper-large-v3-turbo"],
    "ctc": ["jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
            "facebook/wav2vec2-large-960h-lv60-self",
            "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
            "MahmoudAshraf/mms-300m-1130-forced-aligner"],
}
LLM_MODEL = "qwen2.5:7b"
UVR_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def dl_hf(group: str) -> None:
    from huggingface_hub import list_repo_files, snapshot_download

    for repo in HF_REPOS[group]:
        print(f"  -> {repo}", flush=True)
        # skip duplicate weight formats; transformers prefers safetensors over .bin
        ignore = ["*.msgpack", "*.h5", "*.ot", "flax_model*", "tf_model*"]
        if any(f.endswith(".safetensors") for f in list_repo_files(repo)):
            ignore.append("*.bin")
        snapshot_download(repo, ignore_patterns=ignore)


def dl_openai_whisper() -> None:
    import whisper

    root = Path(paths.DIRS["XDG_CACHE_HOME"]) / "whisper"
    for name in ("turbo",):
        print(f"  -> openai-whisper {name}", flush=True)
        whisper._download(whisper._MODELS[name], str(root), False)


def dl_demucs() -> None:
    from demucs.pretrained import get_model

    get_model("htdemucs")


def dl_uvr() -> None:
    import tempfile

    from audio_separator.separator import Separator

    sep = Separator(model_file_dir=str(paths.DIRS["SUBALIGN_UVR_MODEL_DIR"]), output_dir=tempfile.mkdtemp())
    sep.load_model(model_filename=UVR_MODEL)


def dl_funasr() -> None:
    from funasr import AutoModel

    AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc", disable_update=True)


def dl_rnnoise() -> None:
    import urllib.request

    d = paths.MODELS / "rnnoise"
    d.mkdir(parents=True, exist_ok=True)
    for m in ("somnolent-hogwash-2018-09-01/sh.rnnn", "beguiling-drafter-2018-08-30/bd.rnnn"):
        dst = d / m.split("/")[-1]
        if not dst.exists():
            urllib.request.urlretrieve(f"https://github.com/GregorR/rnnoise-models/raw/master/{m}", dst)


def dl_campplus() -> None:
    from funasr import AutoModel

    AutoModel(model="cam++", disable_update=True)          # speaker embeddings for --diarize


def dl_llm() -> None:
    from webui.ollama import ensure_server

    ensure_server()
    subprocess.run([str(paths.OLLAMA_EXE), "pull", LLM_MODEL], check=True)


GROUPS = {
    "whisper": lambda: dl_hf("whisper"),
    "openai-whisper": dl_openai_whisper,
    "ctc": lambda: dl_hf("ctc"),
    "demucs": dl_demucs,
    "uvr": dl_uvr,
    "funasr": dl_funasr,
    "rnnoise": dl_rnnoise,
    "campplus": dl_campplus,
    "llm": dl_llm,
}


def main(argv) -> int:
    todo = argv or list(GROUPS)
    failed = []
    for g in todo:
        t = time.time()
        print(f"== {g}", flush=True)
        try:
            GROUPS[g]()
            print(f"   ok ({time.time() - t:.0f}s)", flush=True)
        except Exception:
            traceback.print_exc()
            failed.append(g)
    print("failed: " + ", ".join(failed) if failed else "all models downloaded", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
