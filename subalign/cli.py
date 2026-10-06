"""Command line interface.

    subalign align    video.mp4 [--script script.txt]           # speech subtitles
    subalign lyrics   song.mp3  [--lyrics lyrics.txt|.lrc]        # song karaoke
    subalign separate song.mp3  [--stems vocals,instrumental]     # vocal separation
    subalign convert  in.ass -f lrc-enhanced,srt --style neon     # convert / restyle
    subalign translate in.srt --to en                             # LLM translation
    subalign proofread in.srt --llm-provider deepseek             # LLM proofreading
    subalign formats                                              # list formats & presets
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__


def _csv(s: Optional[str]) -> Optional[List[str]]:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _add_export(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("export")
    g.add_argument("-o", "--out", help="output directory (default: ./output next to the input)")
    g.add_argument("-f", "--formats", help="comma separated formats (see `subalign formats`)")
    g.add_argument("--layout", default="landscape",
                   help="landscape | portrait | square | WxH, comma separated for several (横屏/竖屏)")
    g.add_argument("--font-size", type=int, help="font size in px (changes the line width limit)")
    g.add_argument("--max-chars", type=int, dest="max_units",
                   help="max row width in half-width units (CJK char = 2), overrides the layout")
    g.add_argument("--max-lines", type=int, help="rows per cue")
    g.add_argument("--style", help="style preset name or JSON style file (ASS output)")
    g.add_argument("--punct", default="keep", choices=["keep", "strip", "space"],
                   help="subtitle punctuation policy")
    g.add_argument("--mono", action="store_true", help="do not write translations into the files")


def _add_llm(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("LLM (translation / proofreading)")
    g.add_argument("--llm-provider", default=None,
                   help="anthropic (default) | openai | deepseek | qwen | moonshot | zhipu | ollama | custom")
    g.add_argument("--llm-model")
    g.add_argument("--llm-base-url")
    g.add_argument("--llm-api-key", help="prefer the provider's environment variable")


def _add_align(p: argparse.ArgumentParser, song: bool) -> None:
    g = p.add_argument_group("alignment")
    if not song:
        g.add_argument("--mode", default="auto", choices=["auto", "speech", "song"])
    g.add_argument("--lang", help="language code (zh, en, ja, ...); auto-detected if omitted")
    g.add_argument("--asr", default="auto", help="auto | faster-whisper | whisper | openai | funasr | none")
    g.add_argument("--asr-model", help="e.g. large-v3, medium, paraformer-zh, whisper-1")
    g.add_argument("--ctc", default="auto", choices=["auto", "on", "off"], help="CTC forced alignment")
    g.add_argument("--ctc-model", help="HuggingFace wav2vec2 CTC model id")
    g.add_argument("--separation", default="auto", help="auto | uvr | demucs | dsp | none (songs)")
    g.add_argument("--separation-model")
    g.add_argument("--device", help="cpu | cuda")
    g.add_argument("--no-refine", action="store_true", help="skip acoustic DP refinement")
    g.add_argument("--no-proofread", action="store_true", help="skip the script-vs-audio diff report")
    g.add_argument("--translate", metavar="LANG", help="translate into LANG with the LLM")
    g.add_argument("--llm-proofread", action="store_true", help="LLM fixes ASR errors (no-script mode)")
    g.add_argument("--context", default="", help="topic / names to help LLM proofreading")
    g.add_argument("--glossary", help='JSON file {"wrong or source term": "correct term"}')
    g.add_argument("--remove-fillers", action="store_true", help="drop 嗯/呃/um/uh ...")
    g = p.add_argument_group("transcript quality (recognition mode)")
    g.add_argument("--verbatim", action="store_true", help="strict verbatim: keep 嗯/呃, repeats and false starts")
    g.add_argument("--cross-check", metavar="BACKEND",
                   help="transcribe again with a second ASR backend and mark every disagreement for review")
    g.add_argument("--diarize", action="store_true", help="label speakers (S1, S2, ...)")
    g.add_argument("--speakers", type=int, help="number of speakers (implies --diarize)")
    g.add_argument("--keep-hallucinations", action="store_true",
                   help="do not remove text the recogniser wrote over music / silence")
    g.add_argument("--title")
    g.add_argument("--artist")
    g.add_argument("--album")


def _export_cfg(a):
    from .pipeline import ExportConfig

    return ExportConfig(formats=_csv(a.formats), layouts=_csv(a.layout) or ["landscape"], style=a.style,
                        punct=a.punct, bilingual=not a.mono, font_size=a.font_size, max_units=a.max_units,
                        max_lines=a.max_lines)


def _llm_cfg(a):
    from .pipeline import LLMConfig

    return LLMConfig(provider=a.llm_provider, model=a.llm_model, base_url=a.llm_base_url, api_key=a.llm_api_key)


def _read_script(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    raw = p.read_bytes()
    if raw[:4] == b"krc1":
        from .formats.lyrics import krc_decrypt

        return krc_decrypt(raw)
    return raw.decode("utf-8-sig", errors="replace")


def cmd_align(a, song: bool = False) -> int:
    from .align.aligner import AlignConfig
    from .pipeline import PipelineConfig, run

    glossary = json.loads(Path(a.glossary).read_text(encoding="utf-8")) if a.glossary else {}
    # names / terms help the recogniser too (Whisper initial prompt)
    hint = "，".join(x for x in [a.context.strip()] + list(dict.fromkeys(glossary.values())) if x)
    cfg = PipelineConfig(
        align=AlignConfig(mode="song" if song else a.mode, language=a.lang, asr_backend=a.asr,
                          asr_model=a.asr_model, ctc=a.ctc, ctc_model=a.ctc_model, separation=a.separation,
                          separation_model=a.separation_model, refine=not a.no_refine,
                          proofread=not a.no_proofread, device=a.device, verbatim=a.verbatim, asr_prompt=hint,
                          screen_hallucinations=not a.keep_hallucinations, cross_check=a.cross_check),
        export=_export_cfg(a), llm=_llm_cfg(a), translate_to=a.translate, llm_proofread=a.llm_proofread,
        proofread_context=a.context, remove_fillers=a.remove_fillers, glossary=glossary,
        diarize=a.diarize or bool(a.speakers), speakers=a.speakers,
        metadata={k: v for k, v in (("title", a.title), ("artist", a.artist), ("album", a.album)) if v})
    script = _read_script(getattr(a, "lyrics", None) if song else a.script)
    res = run(a.audio, script, a.out, cfg)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


def cmd_separate(a) -> int:
    from .separate import separate

    out = a.out or str(Path(a.audio).parent / "output")
    res = separate(a.audio, out, backend=a.backend, stems=_csv(a.stems) or ["vocals", "instrumental"],
                   fmt=a.format, model=a.model, device=a.device)
    print(json.dumps({k: str(v) for k, v in res.items()}, ensure_ascii=False, indent=2))
    return 0


def _load_doc(a):
    from .align.timing import shift
    from .formats import read

    doc = read(a.input, a.input_format)
    if getattr(a, "kind", None):
        doc.kind = a.kind
    elif Path(a.input).suffix.lower() in (".lrc", ".qrc", ".krc", ".yrc", ".ttml"):
        doc.kind = "song"
    if getattr(a, "shift", 0):
        shift(doc, a.shift)
    return doc


def cmd_convert(a) -> int:
    from .pipeline import PipelineConfig, export_all

    doc = _load_doc(a)
    cfg = PipelineConfig(export=_export_cfg(a))
    if not cfg.export.formats:
        print("error: -f/--formats is required for convert", file=sys.stderr)
        return 2
    out = a.out or str(Path(a.input).parent / "output")
    files = export_all(doc, out, Path(a.input).stem.split(".")[0], cfg)
    print("\n".join(str(f) for f in files))
    return 0


def cmd_translate(a) -> int:
    from .pipeline import PipelineConfig, export_all

    doc = _load_doc(a)
    from .translate import translate_document

    glossary = json.loads(Path(a.glossary).read_text(encoding="utf-8")) if a.glossary else None
    translate_document(doc, _llm_cfg(a).client(), a.to, glossary=glossary, style=a.instructions or "")
    cfg = PipelineConfig(export=_export_cfg(a))
    if not cfg.export.formats:
        from .formats import guess_format

        cfg.export.formats = [guess_format(a.input)]
    out = a.out or str(Path(a.input).parent / "output")
    files = export_all(doc, out, Path(a.input).stem.split(".")[0] + f".{a.to}", cfg)
    print("\n".join(str(f) for f in files))
    return 0


def cmd_proofread(a) -> int:
    from .pipeline import PipelineConfig, export_all
    from .proofread import apply_glossary, llm_proofread, remove_fillers

    doc = _load_doc(a)
    before = [ln.text for ln in doc.lines]
    if a.remove_fillers:
        remove_fillers(doc, a.lang or "zh")
    glossary = json.loads(Path(a.glossary).read_text(encoding="utf-8")) if a.glossary else None
    if not a.rules_only:
        llm_proofread(doc, _llm_cfg(a).client(), context=a.context, glossary=glossary)
    if glossary:
        apply_glossary(doc, glossary)
    changed = sum(1 for x, ln in zip(before, doc.lines) if x != ln.text)
    cfg = PipelineConfig(export=_export_cfg(a))
    if not cfg.export.formats:
        from .formats import guess_format

        cfg.export.formats = [guess_format(a.input)]
    out = a.out or str(Path(a.input).parent / "output")
    files = export_all(doc, out, Path(a.input).stem.split(".")[0] + ".proofread", cfg)
    print(f"{changed} line(s) changed")
    print("\n".join(str(f) for f in files))
    return 0


def cmd_roughcut(a) -> int:
    from .pipeline import PipelineConfig, export_all
    from .roughcut import RoughCutConfig, rough_cut

    cfg = RoughCutConfig(level=a.level, fillers=not a.no_fillers, repeats=not a.no_repeats,
                         retakes=not a.no_retakes, unrecognized=not a.no_unrecognized, pauses=not a.no_pauses,
                         max_pause=a.max_pause, keep_pause=a.keep_pause, min_gap=a.min_gap,
                         crossfade_ms=a.crossfade_ms, extra_fillers=_csv(a.fillers) or (),
                         keep_words=_csv(a.keep) or (), llm=a.llm)
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8")) if a.plan else None
    out = a.out or str(Path(a.audio).parent / "output")
    res = rough_cut(Path(a.audio), Path(out), cfg, script=_read_script(a.script), plan=plan, language=a.lang,
                    asr_backend=a.asr, asr_model=a.asr_model, device=a.device, ctc=a.ctc,
                    audio_format=a.format, video=not a.no_video,
                    llm_client=_llm_cfg(a).client() if a.llm and plan is None else None)
    files = [str(f) for f in res["files"]]
    pcfg = PipelineConfig(export=_export_cfg(a))
    if pcfg.export.formats is None:
        pcfg.export.formats = ["srt", "json"]
    if res["document"].lines and pcfg.export.formats:
        files += [str(f) for f in export_all(res["document"], out, res["base"] + ".cut", pcfg)]
    print(json.dumps({"stats": res["stats"], "files": files}, ensure_ascii=False, indent=2))
    return 0


def cmd_videocut(a) -> int:
    from .pipeline import PipelineConfig, export_all
    from .roughcut import RoughCutConfig
    from .videocut import VideoCutConfig, video_cut

    rcfg = RoughCutConfig(level=a.level, fillers=not a.no_fillers, repeats=not a.no_repeats,
                          retakes=not a.no_retakes, unrecognized=not a.no_unrecognized, pauses=not a.no_pauses,
                          max_pause=a.max_pause, keep_pause=a.keep_pause, min_gap=a.min_gap,
                          crossfade_ms=a.crossfade_ms, extra_fillers=_csv(a.fillers) or (),
                          keep_words=_csv(a.keep) or (), llm=a.llm)
    vcfg = VideoCutConfig(min_shot=a.min_shot, max_cuts=a.max_cuts, mute=not a.no_mute, mute_max=a.mute_max,
                          slide=a.slide, transitions=a.transitions, zoom=a.zoom, fade_frames=a.fade_frames,
                          content=a.content, crf=a.crf, fcpxml=not a.no_fcpxml)
    plan = json.loads(Path(a.plan).read_text(encoding="utf-8")) if a.plan else None
    out = a.out or str(Path(a.video).parent / "output")
    res = video_cut(Path(a.video), Path(out), rcfg, vcfg, script=_read_script(a.script), plan=plan, language=a.lang,
                    asr_backend=a.asr, asr_model=a.asr_model, device=a.device, ctc=a.ctc,
                    llm_client=_llm_cfg(a).client() if a.llm and plan is None else None)
    files = [str(f) for f in res["files"]]
    pcfg = PipelineConfig(export=_export_cfg(a))
    if pcfg.export.formats is None:
        pcfg.export.formats = ["srt", "json"]
    if res["document"].lines and pcfg.export.formats:
        files += [str(f) for f in export_all(res["document"], out, res["base"] + ".vcut", pcfg)]
    print(json.dumps({"stats": res["stats"], "files": files}, ensure_ascii=False, indent=2))
    return 0


def cmd_studio(a) -> int:
    from .studio import StudioConfig, process

    cfg = StudioConfig(
        cleanup=a.cleanup, fillers=not a.no_fillers, breaths=a.breaths, breath_reduce_db=a.breath_reduce,
        coughs=not a.keep_coughs, pauses=a.shorten_pauses, extra_fillers=tuple(_csv(a.fillers) or ()), language=a.lang,
        highpass=a.highpass, denoise=a.denoise, declick=not a.no_declick, plosives=not a.no_plosives, deess=a.deess,
        eq=not a.no_eq, mud_gain=a.mud, presence_gain=a.presence, air_gain=a.air,
        compress=not a.no_compress, comp_threshold=a.comp_threshold, comp_ratio=a.comp_ratio,
        reverb=a.reverb, bgm_ratio=a.bgm_ratio, bgm_duck=not a.no_duck,
        loudness=None if a.loudness is not None and a.loudness >= 0 else a.loudness, true_peak=a.true_peak,
        format=a.format, bitrate=a.bitrate, bit_depth=a.bit_depth, out_sr=a.sample_rate, channels=a.channels)
    out = a.out or str(Path(a.audio).parent / "output")
    res = process(Path(a.audio), Path(out), cfg, bgm_path=Path(a.bgm) if a.bgm else None, asr_backend=a.asr,
                  asr_model=a.asr_model, device=a.device)
    print(json.dumps({"files": [str(f) for f in res["files"]], "steps": res["report"]["steps"],
                      "input": res["report"]["input"], "output": res["report"]["output"]}, ensure_ascii=False, indent=2))
    return 0


def cmd_formats(_a) -> int:
    from .formats import FORMATS
    from .segment.layout import LAYOUTS
    from .style import PRESETS

    print("Formats:")
    for f in FORMATS.values():
        rw = "read/write" if f.reader else "write"
        print(f"  {f.name:14s} {f.ext:9s} {f.level:5s} {rw:10s} {f.description}")
    print("\nStyle presets: " + ", ".join(PRESETS))
    print("Layouts: " + ", ".join(f"{k} ({v.play_res_x}x{v.play_res_y}, {v.max_units} units)" for k, v in LAYOUTS.items()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="subalign", description="精准逐字字幕 / 歌词对齐工具")
    p.add_argument("--version", action="version", version=f"subalign {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("align", help="speech (or auto) alignment: ASR or script -> subtitles")
    s.add_argument("audio")
    s.add_argument("--script", help="script/transcript (.txt one line per subtitle, or .srt/.lrc/...)")
    _add_align(s, song=False)
    _add_export(s)
    _add_llm(s)
    s.set_defaults(func=lambda a: cmd_align(a, song=False))

    s = sub.add_parser("lyrics", help="song lyric alignment (char-level + line-level)")
    s.add_argument("audio")
    s.add_argument("--lyrics", help="lyrics file (.txt/.lrc/.qrc/.krc/...); omit to recognise the lyrics")
    _add_align(s, song=True)
    _add_export(s)
    _add_llm(s)
    s.set_defaults(func=lambda a: cmd_align(a, song=True))

    s = sub.add_parser("separate", help="vocal / accompaniment separation")
    s.add_argument("audio")
    s.add_argument("-o", "--out")
    s.add_argument("--backend", default="auto", choices=["auto", "uvr", "demucs", "dsp"])
    s.add_argument("--stems", default="vocals,instrumental", help="vocals, instrumental or both")
    s.add_argument("--format", default="wav", help="wav | flac | mp3 | m4a")
    s.add_argument("--model", help="demucs model (htdemucs, htdemucs_ft) or audio-separator model file")
    s.add_argument("--device")
    s.set_defaults(func=cmd_separate)

    def add_input(sp):
        sp.add_argument("input")
        sp.add_argument("--input-format", help="force the input format")
        sp.add_argument("--kind", choices=["speech", "song"])
        sp.add_argument("--shift", type=float, default=0.0, help="shift all times by seconds")

    s = sub.add_parser("convert", help="convert between formats / re-style / re-layout")
    add_input(s)
    _add_export(s)
    s.set_defaults(func=cmd_convert)

    s = sub.add_parser("translate", help="LLM translation of an existing subtitle / lyric file")
    add_input(s)
    s.add_argument("--to", required=True, help="target language (en, zh, ja, ko, ...)")
    s.add_argument("--glossary")
    s.add_argument("--instructions", help="extra style instructions for the translator")
    _add_export(s)
    _add_llm(s)
    s.set_defaults(func=cmd_translate)

    s = sub.add_parser("proofread", help="LLM / rule-based proofreading of a subtitle file")
    add_input(s)
    s.add_argument("--lang")
    s.add_argument("--context", default="")
    s.add_argument("--glossary")
    s.add_argument("--remove-fillers", action="store_true")
    s.add_argument("--rules-only", action="store_true", help="glossary / filler rules only, no LLM")
    _add_export(s)
    _add_llm(s)
    s.set_defaults(func=cmd_proofread)

    s = sub.add_parser("roughcut", help="speech rough cut: remove fillers / repeats / retakes / long pauses")
    s.add_argument("audio")
    s.add_argument("--script", help="verbatim transcript (optional; recognised otherwise)")
    s.add_argument("--plan", help="edited plan (.roughcut.json) to render instead of detecting again")
    g = s.add_argument_group("recognition")
    g.add_argument("--lang")
    g.add_argument("--asr", default="auto", help="auto | faster-whisper | whisper | openai | funasr")
    g.add_argument("--asr-model")
    g.add_argument("--ctc", default="auto", choices=["auto", "on", "off"], help="CTC (only with --script)")
    g.add_argument("--device")
    g = s.add_argument_group("what to cut")
    g.add_argument("--level", default="standard", choices=["conservative", "standard", "aggressive"])
    g.add_argument("--no-fillers", action="store_true", help="keep 嗯/呃/um ...")
    g.add_argument("--no-repeats", action="store_true", help="keep stutters / repeated words")
    g.add_argument("--no-retakes", action="store_true", help="keep sentences said twice")
    g.add_argument("--no-unrecognized", action="store_true", help="keep voiced sounds the recogniser skipped")
    g.add_argument("--no-pauses", action="store_true", help="do not shorten long pauses")
    g.add_argument("--max-pause", type=float, help="pauses longer than this are shortened (s)")
    g.add_argument("--keep-pause", type=float, help="length a long pause is shortened to (s)")
    g.add_argument("--min-gap", type=float, default=0.12, help="pause left where a cut joins two words (s)")
    g.add_argument("--crossfade-ms", type=float, default=25.0)
    g.add_argument("--fillers", help="extra filler words, comma separated")
    g.add_argument("--keep", help="words never cut, comma separated")
    g.add_argument("--llm", action="store_true", help="let the LLM mark redundant phrases (废话)")
    g.add_argument("--format", help="output audio format: wav | mp3 | flac | m4a (default: like the input)")
    g.add_argument("--no-video", action="store_true", help="audio only, even for video input")
    _add_export(s)
    _add_llm(s)
    s.set_defaults(func=cmd_roughcut)

    s = sub.add_parser("videocut", help="video rough cut: speech rough cut with as few, well-placed picture cuts")
    s.add_argument("video")
    s.add_argument("--script", help="verbatim transcript (optional; recognised otherwise)")
    s.add_argument("--plan", help="edited plan (.videocut.json) to render instead of detecting again")
    g = s.add_argument_group("recognition")
    g.add_argument("--lang")
    g.add_argument("--asr", default="auto", help="auto | faster-whisper | whisper | openai | funasr")
    g.add_argument("--asr-model")
    g.add_argument("--ctc", default="auto", choices=["auto", "on", "off"], help="CTC (only with --script)")
    g.add_argument("--device")
    g = s.add_argument_group("what to cut (as the speech rough cut)")
    g.add_argument("--level", default="standard", choices=["conservative", "standard", "aggressive"])
    g.add_argument("--no-fillers", action="store_true")
    g.add_argument("--no-repeats", action="store_true")
    g.add_argument("--no-retakes", action="store_true")
    g.add_argument("--no-unrecognized", action="store_true")
    g.add_argument("--no-pauses", action="store_true")
    g.add_argument("--max-pause", type=float)
    g.add_argument("--keep-pause", type=float)
    g.add_argument("--min-gap", type=float, default=0.12)
    g.add_argument("--crossfade-ms", type=float, default=25.0)
    g.add_argument("--fillers", help="extra filler words, comma separated")
    g.add_argument("--keep", help="words never cut, comma separated")
    g.add_argument("--llm", action="store_true", help="let the LLM mark redundant phrases (废话)")
    g = s.add_argument_group("picture")
    g.add_argument("--min-shot", type=float, default=1.5, help="shortest picture segment between two cuts (s)")
    g.add_argument("--max-cuts", type=int, default=3, help="at most this many picture cuts in any 10 s")
    g.add_argument("--no-mute", action="store_true", help="never silence a filler instead of cutting it")
    g.add_argument("--mute-max", type=float, default=0.45, help="longest filler that may be silenced (s)")
    g.add_argument("--slide", type=float, default=0.25, help="how far a picture cut may move inside a pause (s)")
    g.add_argument("--transitions", default="auto", choices=["auto", "cut", "fade", "zoom"])
    g.add_argument("--zoom", type=float, default=1.12, help="punch-in scale")
    g.add_argument("--fade-frames", type=int, default=4)
    g.add_argument("--content", default="auto", choices=["auto", "talking", "screen"])
    g.add_argument("--crf", type=int, default=20)
    g.add_argument("--no-fcpxml", action="store_true")
    _add_export(s)
    _add_llm(s)
    s.set_defaults(func=cmd_videocut)

    s = sub.add_parser("studio", help="voice-over processing: cleanup, repair, EQ, dynamics, BGM, loudness, export")
    s.add_argument("audio")
    s.add_argument("-o", "--out")
    s.add_argument("--bgm", help="background music file mixed under the voice")
    g = s.add_argument_group("cleanup (edits the timeline)")
    g.add_argument("--cleanup", action="store_true", help="remove breaths / coughs / fillers")
    g.add_argument("--no-fillers", action="store_true", help="cleanup without recognition (breaths / coughs only)")
    g.add_argument("--breaths", default="reduce", choices=["off", "reduce", "remove"])
    g.add_argument("--breath-reduce", type=float, default=12.0, help="dB a breath is turned down by")
    g.add_argument("--keep-coughs", action="store_true")
    g.add_argument("--shorten-pauses", action="store_true")
    g.add_argument("--fillers", help="extra filler words, comma separated")
    g.add_argument("--lang")
    g.add_argument("--asr", default="auto")
    g.add_argument("--asr-model")
    g.add_argument("--device")
    g = s.add_argument_group("repair / tone / dynamics")
    g.add_argument("--highpass", type=float, default=80.0, help="low cut (Hz), 0 = off")
    g.add_argument("--denoise", default="medium", choices=["off", "light", "medium", "strong"])
    g.add_argument("--no-declick", action="store_true")
    g.add_argument("--no-plosives", action="store_true")
    g.add_argument("--deess", type=float, default=6.0, help="max de-ess reduction (dB), 0 = off")
    g.add_argument("--no-eq", action="store_true")
    g.add_argument("--mud", type=float, default=-3.0, help="dB at ~250 Hz")
    g.add_argument("--presence", type=float, default=2.5, help="dB at ~3 kHz")
    g.add_argument("--air", type=float, default=0.0, help="dB high shelf at 10 kHz")
    g.add_argument("--no-compress", action="store_true")
    g.add_argument("--comp-threshold", type=float, default=-20.0)
    g.add_argument("--comp-ratio", type=float, default=3.0)
    g.add_argument("--reverb", type=float, default=0.0, help="wet amount 0..1 (0.1-0.2 = light)")
    g.add_argument("--bgm-ratio", type=float, default=0.2, help="BGM loudness relative to the voice (0.15-0.25)")
    g.add_argument("--no-duck", action="store_true", help="no extra BGM dip under speech")
    g = s.add_argument_group("loudness / export")
    g.add_argument("--loudness", type=float, default=-16.0, help="target LUFS (e.g. -14, -16, -18, -23); 0 = off")
    g.add_argument("--true-peak", type=float, default=-1.5)
    g.add_argument("--format", default="wav", choices=["wav", "mp3", "aac", "flac", "opus"])
    g.add_argument("--bitrate", default="320k")
    g.add_argument("--bit-depth", type=int, default=24, choices=[16, 24, 32])
    g.add_argument("--sample-rate", type=int, default=48000)
    g.add_argument("--channels", type=int, default=2, choices=[1, 2])
    s.set_defaults(func=cmd_studio)

    s = sub.add_parser("formats", help="list formats, style presets and layouts")
    s.set_defaults(func=cmd_formats)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return a.func(a)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
