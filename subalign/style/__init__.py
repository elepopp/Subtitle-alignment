"""Subtitle styling: ASS styles, karaoke / per-syllable effects and presets.

A style config can be given as a preset name or a JSON file::

    {
      "preset": "karaoke",
      "main":        {"fontname": "思源黑体", "fontsize": 72, "primary": "#FFD700"},
      "translation": {"fontsize": 44},
      "effect":      {"karaoke": "kf", "syllable": "pop", "fade_in_ms": 150}
    }

Colours are ``#RRGGBB`` or ``#RRGGBBAA`` (AA = opacity, FF = opaque) or raw
ASS ``&HAABBGGRR``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..segment.layout import Layout


@dataclass
class AssStyle:
    name: str = "Default"
    fontname: str = "Noto Sans CJK SC"
    fontsize: int = 64
    primary: str = "#FFFFFF"        # sung / normal text colour
    secondary: str = "#FFFFFF"      # unsung colour for karaoke
    outline_color: str = "#000000"
    back_color: str = "#00000080"
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strikeout: bool = False
    scale_x: float = 100
    scale_y: float = 100
    spacing: float = 0
    angle: float = 0
    border_style: int = 1           # 1 outline+shadow, 3 opaque box
    outline: float = 3
    shadow: float = 1
    alignment: int = 2              # numpad layout, 2 = bottom centre
    margin_l: int = 60
    margin_r: int = 60
    margin_v: int = 60
    encoding: int = 1


@dataclass
class Effect:
    karaoke: str = "none"           # none | k | kf | ko
    syllable: str = "none"          # none | pop | bounce | glow | typewriter
    fade_in_ms: int = 0
    fade_out_ms: int = 0
    lead_in_ms: int = 0             # show the line this long before it is sung
    lead_out_ms: int = 0            # keep it this long after
    pop_scale: int = 118
    glow_color: str = "#FFF3A0"
    glow_blur: float = 6
    base_blur: float = 0.6


@dataclass
class StyleConfig:
    main: AssStyle = field(default_factory=AssStyle)
    translation: AssStyle = field(default_factory=lambda: AssStyle(name="Translation", fontsize=44))
    effect: Effect = field(default_factory=Effect)
    translation_position: str = "below"   # below | above

    def adapt_to_layout(self, layout: Layout) -> "StyleConfig":
        """Scale font sizes / margins to the target resolution."""
        cfg = replace(self, main=replace(self.main), translation=replace(self.translation))
        ratio = cfg.translation.fontsize / max(1, cfg.main.fontsize)
        cfg.main.fontsize = layout.font_size
        cfg.translation.fontsize = max(12, int(round(layout.font_size * ratio)))
        scale = min(layout.play_res_x, layout.play_res_y) / 1080.0
        for st in (cfg.main, cfg.translation):
            st.margin_l = st.margin_r = int(round(layout.play_res_x * (1 - layout.safe_area) / 2))
            st.outline = round(st.outline * scale, 1)
            st.shadow = round(st.shadow * scale, 1)
        cfg.main.margin_v = layout.margin_v
        gap = int(round(cfg.translation.fontsize * 1.25))
        if cfg.translation_position == "below":
            cfg.translation.margin_v = max(0, layout.margin_v - gap)
            cfg.main.margin_v = layout.margin_v + (gap if layout.margin_v - gap < 10 else 0)
        else:
            cfg.translation.margin_v = layout.margin_v + int(round(cfg.main.fontsize * 1.25))
        return cfg


PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {},
    "box": {"main": {"border_style": 3, "outline": 8, "shadow": 0, "outline_color": "#00000099",
                     "back_color": "#00000099"}},
    "karaoke": {"main": {"primary": "#FFD34D", "secondary": "#FFFFFF", "outline_color": "#202020",
                         "bold": True, "outline": 3.5, "shadow": 1.5},
                "effect": {"karaoke": "kf", "lead_in_ms": 600, "lead_out_ms": 300,
                           "fade_in_ms": 150, "fade_out_ms": 150}},
    "karaoke-pop": {"main": {"primary": "#FF7AB6", "secondary": "#FFFFFF", "outline_color": "#3A0A26",
                             "bold": True, "outline": 3.5},
                    "effect": {"karaoke": "k", "syllable": "pop", "lead_in_ms": 500, "lead_out_ms": 300,
                               "fade_in_ms": 120, "fade_out_ms": 120}},
    "neon": {"main": {"primary": "#7DF9FF", "secondary": "#FFFFFF", "outline_color": "#00B7FF",
                      "outline": 2.5, "shadow": 0},
             "effect": {"karaoke": "kf", "syllable": "glow", "glow_color": "#00E5FF", "glow_blur": 8,
                        "lead_in_ms": 500, "lead_out_ms": 300, "fade_in_ms": 200, "fade_out_ms": 200}},
    "typewriter": {"effect": {"syllable": "typewriter", "lead_in_ms": 0, "lead_out_ms": 400,
                              "fade_out_ms": 150}},
    "bounce": {"main": {"bold": True}, "effect": {"syllable": "bounce", "karaoke": "k", "lead_in_ms": 300, "lead_out_ms": 300,
                                                "fade_out_ms": 150}},
    # big bold yellow captions popular on short-video platforms (竖屏)
    "shortvideo": {"main": {"primary": "#FFE600", "secondary": "#FFFFFF", "outline_color": "#000000",
                            "bold": True, "outline": 5, "shadow": 0},
                   "translation": {"primary": "#FFFFFF", "outline": 3},
                   "effect": {"syllable": "pop", "karaoke": "k", "pop_scale": 112}},
}


def _apply(obj, overrides: Dict[str, Any]):
    names = {f.name for f in fields(obj)}
    unknown = set(overrides) - names
    if unknown:
        raise ValueError(f"unknown style keys for {type(obj).__name__}: {sorted(unknown)}")
    return replace(obj, **overrides)


def make_style(preset: str = "default", overrides: Optional[Dict[str, Any]] = None) -> StyleConfig:
    if preset not in PRESETS:
        raise ValueError(f"unknown style preset {preset!r}; available: {sorted(PRESETS)}")
    cfg = StyleConfig()
    for src in (PRESETS[preset], overrides or {}):
        if "main" in src:
            cfg.main = _apply(cfg.main, src["main"])
        if "translation" in src:
            cfg.translation = _apply(cfg.translation, src["translation"])
        if "effect" in src:
            cfg.effect = _apply(cfg.effect, src["effect"])
        if "translation_position" in src:
            cfg.translation_position = src["translation_position"]
    cfg.main.name = "Default"
    cfg.translation.name = "Translation"
    return cfg


def load_style(spec: Union[str, Path, None]) -> StyleConfig:
    """Preset name, or path to a JSON style file (optionally with "preset")."""
    if spec is None:
        return make_style()
    p = Path(str(spec))
    if p.suffix.lower() == ".json" and p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        return make_style(data.pop("preset", "default"), data)
    return make_style(str(spec))


def style_to_dict(cfg: StyleConfig) -> Dict[str, Any]:
    return asdict(cfg)


def ass_color(c: str) -> str:
    """#RRGGBB[AA] (AA = opacity) -> &HAABBGGRR (AA = transparency)."""
    c = c.strip()
    if c.upper().startswith("&H"):
        return c.upper().rstrip("&")
    c = c.lstrip("#")
    if len(c) not in (6, 8):
        raise ValueError(f"bad colour {c!r}")
    r, g, b = c[0:2], c[2:4], c[4:6]
    a = 0xFF - int(c[6:8], 16) if len(c) == 8 else 0
    return f"&H{a:02X}{b}{g}{r}".upper()


def ass_color_tag(c: str) -> str:
    """Colour for override tags (\\1c etc.): &HBBGGRR&."""
    full = ass_color(c)
    return "&H" + full[-6:] + "&"
