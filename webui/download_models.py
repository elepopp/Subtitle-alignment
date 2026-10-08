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
LLM_MODEL = "index-translate:9b-q5"
# not in the Ollama registry: the GGUF comes from ModelScope and is imported with this Modelfile.
# Ollama does not convert the GGUF's jinja chat template, so it is spelled out (thinking off)
LLM_GGUF = ("IndexTeam/Index-Translate-9B-GGUF", "Index-Translate-9B.Q5_K_M.gguf")
LLM_MODELFILE = '''FROM ./{gguf}
TEMPLATE """{{{{- if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
<think>

</think>

{{{{ .Response }}}}"""
PARAMETER stop <|im_end|>
PARAMETER stop <|im_start|>
PARAMETER temperature 0
'''
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
    import shutil

    from modelscope.hub.file_download import model_file_download

    from webui.ollama import ensure_server, models

    ensure_server()
    if LLM_MODEL in models():
        return
    tmp = paths.MODELS / "gguf"
    repo, name = LLM_GGUF
    model_file_download(repo, name, local_dir=str(tmp))
    (tmp / "Modelfile").write_text(LLM_MODELFILE.format(gguf=name), encoding="utf-8")
    subprocess.run([str(paths.OLLAMA_EXE), "create", LLM_MODEL, "-f", "Modelfile"], cwd=tmp, check=True)
    shutil.rmtree(tmp)                                     # Ollama keeps its own copy in models/ollama


def dl_indextts() -> None:
    """IndexTTS-2.5 weights (AI 配音); its helper models (w2v-bert, codec, campplus,
    BigVGAN) are fetched by the worker on first load into models/indextts-2.5/hf_cache."""
    from huggingface_hub import hf_hub_download
    from modelscope import snapshot_download

    d = paths.MODELS / "indextts-2.5"
    snapshot_download("IndexTeam/IndexTTS-2.5", local_dir=str(d))
    # the vocoder is not on ModelScope (and the worker's hf-mirror fallback is unreliable);
    # large files over the proxy often break mid-transfer, the download resumes
    for name in ("config.json", "bigvgan_generator.pt"):
        for attempt in range(8):
            try:
                hf_hub_download("nvidia/bigvgan_v2_22khz_80band_256x", name, local_dir=str(d / "hf_cache" / "bigvgan"))
                break
            except Exception:
                if attempt == 7:
                    raise


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
    "indextts": dl_indextts,
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
