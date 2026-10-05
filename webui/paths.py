"""Keep every model / cache inside the project folder.

Import this module before anything that may download a model.  Only
environment variables are set, so subprocesses (demucs) inherit them too.

    models/huggingface   faster-whisper, wav2vec2 / MMS CTC   (HF_HOME)
    models/cache/whisper openai-whisper                       (XDG_CACHE_HOME)
    models/torch         demucs                               (TORCH_HOME)
    models/modelscope    FunASR Paraformer                    (MODELSCOPE_CACHE)
    models/audio-separator  BS-RoFormer                       (SUBALIGN_UVR_MODEL_DIR)
    models/ollama        local LLM                            (OLLAMA_MODELS)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
WORK = ROOT / "webui_data"
OLLAMA_EXE = ROOT / "tools" / "ollama" / "ollama.exe"

DIRS = {
    "HF_HOME": MODELS / "huggingface",
    "XDG_CACHE_HOME": MODELS / "cache",
    "TORCH_HOME": MODELS / "torch",
    "MODELSCOPE_CACHE": MODELS / "modelscope",
    "SUBALIGN_UVR_MODEL_DIR": MODELS / "audio-separator",
    "OLLAMA_MODELS": MODELS / "ollama",
}


def setup() -> None:
    for k, v in DIRS.items():
        v.mkdir(parents=True, exist_ok=True)
        os.environ[k] = str(v)
    # downloads go through the system proxy; set HF_ENDPOINT yourself to use a mirror
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # ctranslate2 (faster-whisper) needs cuBLAS / cuDNN DLLs: reuse the ones bundled with torch
    if sys.platform == "win32":
        try:
            import importlib.util

            spec = importlib.util.find_spec("torch")
            if spec and spec.origin:
                lib = Path(spec.origin).parent / "lib"
                if lib.is_dir():
                    os.add_dll_directory(str(lib))
                    os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            pass
    WORK.mkdir(parents=True, exist_ok=True)


setup()
