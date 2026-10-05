"""Portable Ollama (tools/ollama) serving models from models/ollama."""
from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request

from . import paths

URL = "http://127.0.0.1:11434"


def running() -> bool:
    try:
        with urllib.request.urlopen(URL + "/api/version", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def models() -> list:
    try:
        with urllib.request.urlopen(URL + "/api/tags", timeout=3) as r:
            return [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
    except Exception:
        return []


def ensure_server() -> bool:
    if running():
        return True
    if not paths.OLLAMA_EXE.exists():
        return False
    log = open(paths.WORK / "ollama.log", "ab")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    subprocess.Popen([str(paths.OLLAMA_EXE), "serve"], stdout=log, stderr=log, env=os.environ.copy(),
                     creationflags=flags)
    for _ in range(60):
        if running():
            return True
        time.sleep(0.5)
    return False
