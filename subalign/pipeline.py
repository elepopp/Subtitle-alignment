"""End-to-end pipeline: align -> proofread -> line breaking -> translate -> export."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

from .align.aligner import AlignConfig, AlignResult, align_audio
from .formats import FORMATS, save
from .models import Document
from .segment.layout import Layout, get_layout
from .segment.linebreak import segment_document
from .style import StyleConfig, load_style

log = logging.getLogger("subalign")

DEFAULT_FORMATS = {"speech": ["srt", "ass", "vtt", "json"],
                   "song": ["lrc", "lrc-enhanced", "ass", "srt", "qrc", "krc", "yrc", "ttml", "json"]}


@dataclass
class ExportConfig:
    formats: Optional[Sequence[str]] = None
    layouts: Sequence[str] = ("landscape",)
    style: Union[str, StyleConfig, None] = None
    punct: str = "keep"                 # keep | strip | space
    bilingual: bool = True
    font_size: Optional[int] = None
    max_units: Optional[int] = None
    max_lines: Optional[int] = None


@dataclass
class LLMConfig:
    provider: Optional[str] = None      # anthropic | openai | deepseek | qwen | ...
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None

    def client(self):
        from .llm import make_client

        return make_client(self.provider or "anthropic", self.model, self.base_url, self.api_key)


@dataclass
class PipelineConfig:
    align: AlignConfig = field(default_factory=AlignConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    translate_to: Optional[str] = None
    llm_proofread: bool = False
    proofread_context: str = ""
    glossary: Dict[str, str] = field(default_factory=dict)
    remove_fillers: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)


def postprocess(doc: Document, cfg: PipelineConfig) -> Document:
    from .proofread import apply_glossary, llm_proofread, remove_fillers

    if cfg.remove_fillers:
        remove_fillers(doc, doc.language or "zh")
    if cfg.llm_proofread:
        llm_proofread(doc, cfg.llm.client(), context=cfg.proofread_context, glossary=cfg.glossary or None)
    if cfg.glossary:
        apply_glossary(doc, cfg.glossary)
    return doc


def export_all(doc: Document, out_dir: Union[str, Path], basename: str, cfg: PipelineConfig,
               translate_client=None) -> List[Path]:
    """Segment per layout, optionally translate, and write every format."""
    ex = cfg.export
    formats = list(ex.formats or DEFAULT_FORMATS.get(doc.kind, DEFAULT_FORMATS["speech"]))
    for f in formats:
        if f not in FORMATS:
            raise ValueError(f"unknown format {f!r}")
    style = ex.style if isinstance(ex.style, StyleConfig) else load_style(
        ex.style or ("karaoke" if doc.kind == "song" else "default"))
    out_dir = Path(out_dir)
    written: List[Path] = []
    multi = len(ex.layouts) > 1
    for lay_name in ex.layouts:
        layout = get_layout(lay_name, ex.font_size, ex.max_units, ex.max_lines)
        seg = segment_document(doc, layout, punct=ex.punct)
        if cfg.translate_to and translate_client is not None and not all(ln.translation for ln in seg.lines):
            from .translate import translate_document

            translate_document(seg, translate_client, cfg.translate_to, glossary=cfg.glossary or None,
                               max_units=layout.max_units)
        suffix = f".{layout.name}" if multi else ""
        for f in formats:
            fmt = FORMATS[f]
            # variants sharing an extension get a distinguishing infix: song.enhanced.lrc
            variant = f.split("-", 1)[1] if "-" in f and f != "krc-encrypted" else ""
            path = out_dir / f"{basename}{suffix}{'.' + variant if variant else ''}{fmt.ext}"
            opts = dict(bilingual=ex.bilingual, punct=ex.punct)
            if f == "ass":
                opts.update(style=style, layout=layout)
            save(seg, path, fmt=f, **opts)
            written.append(path)
    return written


def run(audio: Union[str, Path], script: Optional[str] = None, out_dir: Union[str, Path, None] = None,
        cfg: Optional[PipelineConfig] = None) -> Dict:
    cfg = cfg or PipelineConfig()
    audio = Path(audio)
    out_dir = Path(out_dir) if out_dir else audio.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    res: AlignResult = align_audio(audio, script, cfg.align, workdir=out_dir / "work")
    doc = res.document
    doc.metadata.update(cfg.metadata)
    postprocess(doc, cfg)
    client = cfg.llm.client() if cfg.translate_to else None
    files = export_all(doc, out_dir, audio.stem, cfg, translate_client=client)
    if res.report is not None:
        from .proofread import format_report

        rp = out_dir / f"{audio.stem}.proofread.md"
        rp.write_text(format_report(res.report), encoding="utf-8")
        (out_dir / f"{audio.stem}.proofread.json").write_text(
            json.dumps(res.report, ensure_ascii=False, indent=2), encoding="utf-8")
        files.append(rp)
    return {"mode": res.mode, "anchors": res.anchor_source, "files": [str(f) for f in files],
            "stems": {k: str(v) for k, v in res.stems.items()}, "lines": len(doc.lines)}
