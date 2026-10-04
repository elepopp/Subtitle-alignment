"""Advanced SubStation Alpha (.ass) writer / reader with karaoke and
per-syllable effects."""
from __future__ import annotations

import re
from typing import List, Optional

from ..models import Document, Line
from ..segment.layout import Layout, get_layout
from ..style import AssStyle, StyleConfig, ass_color, ass_color_tag, make_style
from ..text.tokenize import make_line
from .common import line_text, parse_ts, strip_tags, token_display, tokens_with_times, ts_ass

STYLE_FORMAT = ("Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
                "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
                "Alignment, MarginL, MarginR, MarginV, Encoding")
EVENT_FORMAT = "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


def _style_line(s: AssStyle) -> str:
    b = lambda v: "-1" if v else "0"  # noqa: E731
    return (f"Style: {s.name},{s.fontname},{s.fontsize},{ass_color(s.primary)},{ass_color(s.secondary)},"
            f"{ass_color(s.outline_color)},{ass_color(s.back_color)},{b(s.bold)},{b(s.italic)},{b(s.underline)},"
            f"{b(s.strikeout)},{s.scale_x:g},{s.scale_y:g},{s.spacing:g},{s.angle:g},{s.border_style},"
            f"{s.outline:g},{s.shadow:g},{s.alignment},{s.margin_l},{s.margin_r},{s.margin_v},{s.encoding}")


def _escape(s: str) -> str:
    return s.replace("\\", "\uff3c").replace("{", "｛").replace("}", "｝").replace("\n", "\\N")


def _syllable_tags(cfg: StyleConfig, a_ms: int, e_ms: int) -> str:
    """Per-syllable override tags.  Every animated property is reset statically
    first so that the previous syllable's transform does not leak into this one."""
    ef = cfg.effect
    kind = ef.syllable
    if kind == "pop":
        s = ef.pop_scale
        return (f"\\fscx100\\fscy100\\t({a_ms},{a_ms + 80},\\fscx{s}\\fscy{s})"
                f"\\t({a_ms + 80},{a_ms + 240},\\fscx100\\fscy100)")
    if kind == "bounce":
        return (f"\\fscy100\\t({a_ms},{a_ms + 90},\\fscy{ef.pop_scale + 12})"
                f"\\t({a_ms + 90},{a_ms + 230},\\fscy100)")
    if kind == "glow":
        base = ass_color_tag(cfg.main.outline_color)
        glow = ass_color_tag(ef.glow_color)
        return (f"\\3c{base}\\blur{ef.base_blur:g}\\t({a_ms},{a_ms + 60},\\3c{glow}\\blur{ef.glow_blur:g})"
                f"\\t({max(e_ms, a_ms + 60)},{max(e_ms, a_ms + 60) + 250},\\3c{base}\\blur{ef.base_blur:g})")
    if kind == "typewriter":
        return f"\\alpha&HFF&\\t({a_ms},{a_ms + 1},\\alpha&H00&)"
    return ""


def karaoke_text(ln: Line, cfg: StyleConfig, show_start: float, punct: str = "keep") -> str:
    ef = cfg.effect
    kt = {"k": "\\k", "kf": "\\kf", "ko": "\\ko"}.get(ef.karaoke)
    if not kt and ef.syllable == "none":
        return _escape(line_text(ln, punct, sep="\n"))
    toks = ln.tokens
    parts: List[str] = []
    cursor_cs = 0                              # karaoke clock (cs) relative to show_start
    lead = int(round((ln.tokens[0].start - show_start) * 100)) if toks and toks[0].start is not None else 0
    if kt and lead > 0:
        parts.append(f"{{{kt}{lead}}}")
        cursor_cs = lead
    for i, t in enumerate(toks):
        a = t.start if t.start is not None else show_start
        e = t.end if t.end is not None else a
        nxt = toks[i + 1].start if i + 1 < len(toks) and toks[i + 1].start is not None else e
        # karaoke boundaries are cumulative so rounding never drifts
        a_cs = int(round((a - show_start) * 100))
        e_cs = int(round((min(e, nxt) - show_start) * 100))
        n_cs = int(round((nxt - show_start) * 100))
        tags = ""
        if kt:
            if a_cs > cursor_cs:          # unexpected gap before the syllable
                parts.append(f"{{{kt}{a_cs - cursor_cs}}}")
                cursor_cs = a_cs
            tags += f"{kt}{max(0, e_cs - cursor_cs)}"
            cursor_cs = max(cursor_cs, e_cs)
        tags += _syllable_tags(cfg, int(round((a - show_start) * 1000)), int(round((e - show_start) * 1000)))
        disp = token_display(toks, i)
        if i in ln.breaks and i < len(toks) - 1:
            disp = disp.rstrip() + "\n"
        parts.append(f"{{{tags}}}{_escape(disp)}")
        if kt and n_cs - cursor_cs >= 5 and i + 1 < len(toks):   # pause inside the line
            parts.append(f"{{{kt}{n_cs - cursor_cs}}}")
            cursor_cs = n_cs
    return "".join(parts)


def write_ass(doc: Document, style: Optional[StyleConfig] = None, layout: Optional[Layout] = None,
              bilingual: bool = True, punct: str = "keep", title: str = "", adapt: bool = True, **_) -> str:
    layout = layout or get_layout("landscape")
    cfg = style or make_style("karaoke" if doc.kind == "song" else "default")
    if adapt:
        cfg = cfg.adapt_to_layout(layout)
    out = ["[Script Info]", f"Title: {title or doc.metadata.get('title', 'subalign')}",
           "ScriptType: v4.00+", "WrapStyle: 2", "ScaledBorderAndShadow: yes", "YCbCr Matrix: TV.709",
           f"PlayResX: {layout.play_res_x}", f"PlayResY: {layout.play_res_y}", "",
           "[V4+ Styles]", f"Format: {STYLE_FORMAT}", _style_line(cfg.main), _style_line(cfg.translation), "",
           "[Events]", f"Format: {EVENT_FORMAT}"]
    ef = cfg.effect
    lines = [ln for ln in doc.lines if ln.tokens and ln.start is not None and ln.end is not None]
    for i, ln in enumerate(lines):
        prev_end = lines[i - 1].end if i > 0 else 0.0
        nxt = lines[i + 1].start if i + 1 < len(lines) else None
        show = max(prev_end if ef.lead_in_ms and ef.karaoke != "none" else 0.0, ln.start - ef.lead_in_ms / 1000)
        show = min(show, ln.start)
        hide = ln.end + ef.lead_out_ms / 1000
        if nxt is not None:
            hide = min(hide, max(ln.end, nxt))
        fade = f"{{\\fad({ef.fade_in_ms},{ef.fade_out_ms})}}" if (ef.fade_in_ms or ef.fade_out_ms) else ""
        body = fade + karaoke_text(ln, cfg, show, punct)
        out.append(f"Dialogue: 0,{ts_ass(show)},{ts_ass(hide)},{ln.style or 'Default'},{ln.speaker or ''},"
                   f"0,0,0,,{body}")
        if bilingual and ln.translation:
            out.append(f"Dialogue: 0,{ts_ass(show)},{ts_ass(hide)},Translation,,0,0,0,,{fade}{_escape(ln.translation)}")
    return "\n".join(out) + "\n"


_K_RE = re.compile(r"\\(?:k|K|kf|ko)(\d+)")


def read_ass(text: str) -> Document:
    """Reads Dialogue events; \\k karaoke timings become token timings.  A
    'Translation'-styled (or same-timed second) event becomes the line's translation."""
    fmt: Optional[List[str]] = None
    lines: List[Line] = []
    styles_seen = []
    in_events = False
    for raw in text.replace("\r\n", "\n").lstrip("\ufeff").split("\n"):
        s = raw.strip()
        if s.startswith("["):
            in_events = s.lower() == "[events]"
            continue
        if not in_events:
            continue
        if s.startswith("Format:"):
            fmt = [f.strip().lower() for f in s[7:].split(",")]
            continue
        if not s.startswith("Dialogue:") or fmt is None:
            continue
        vals = s[9:].split(",", len(fmt) - 1)
        row = dict(zip(fmt, (v.strip() if k != "text" else v for k, v in zip(fmt, vals))))
        a, b = parse_ts(row["start"]), parse_ts(row["end"])
        style = row.get("style", "Default")
        txt = row.get("text", "").replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
        is_trans = "trans" in style.lower() or (lines and abs(lines[-1].start_show - a) < 1e-3
                                                and abs(lines[-1].end_show - b) < 1e-3
                                                and style != lines[-1].style)
        if is_trans and lines:
            lines[-1].translation = strip_tags(txt).strip()
            continue
        if _K_RE.search(txt):
            texts, starts, ends = [], [], []
            t = a
            for m in re.finditer(r"\{([^}]*)\}([^{]*)", txt):
                k = _K_RE.findall(m.group(1))
                dur = sum(int(x) for x in k) / 100 if k else 0.0
                chunk = m.group(2)
                if chunk:
                    texts.append(chunk)
                    starts.append(t)
                    ends.append(t + dur)
                t += dur
            ln = Line(tokens=tokens_with_times(texts, starts, ends))
            ln.update_bounds()
        else:
            ln = make_line(strip_tags(txt).strip(), start=a, end=b)
        ln.style = style if style != "Default" else None
        ln.start_show, ln.end_show = a, b  # type: ignore[attr-defined]
        if ln.tokens:
            lines.append(ln)
            styles_seen.append(style)
    for ln in lines:
        for attr in ("start_show", "end_show"):
            if hasattr(ln, attr):
                delattr(ln, attr)
    return Document(lines=lines)
