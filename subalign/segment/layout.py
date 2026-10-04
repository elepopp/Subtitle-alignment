"""Screen layouts (landscape / portrait / square) and derived line limits.

Width is measured in half-width units: a CJK character counts 2, a Latin
letter 1 (average Latin glyph ~0.5 em).  With a font size ``F`` px a unit is
about ``F/2`` px, so the usable width in units is::

    max_units = play_res_x * safe_area / (font_size / 2)
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass
class Layout:
    name: str = "landscape"
    play_res_x: int = 1920
    play_res_y: int = 1080
    font_size: int = 64
    safe_area: float = 0.84        # fraction of the width usable by text
    max_lines: int = 1             # rows per cue (1 is the norm for CJK; 2 for Latin)
    margin_v: int = 60             # distance from the bottom edge (px)
    max_units_override: Optional[int] = None
    # timing constraints for speech cues
    min_duration: float = 0.7
    max_duration: float = 7.0
    max_cps: float = 17.0          # characters (CJK) / ~ 2 Latin letters per second

    @property
    def max_units(self) -> int:
        if self.max_units_override:
            return self.max_units_override
        return max(8, int(self.play_res_x * self.safe_area / (self.font_size / 2.0)))

    @property
    def orientation(self) -> str:
        if self.play_res_x > self.play_res_y * 1.1:
            return "landscape"
        if self.play_res_y > self.play_res_x * 1.1:
            return "portrait"
        return "square"


LAYOUTS = {
    # 1920x1080, 64px -> ~50 units (25 CJK chars) per row
    "landscape": Layout(),
    # 1080x1920 (抖音/快手/Reels/Shorts): 64px -> ~28 units (14 CJK chars); text sits
    # higher to stay clear of the platform UI at the bottom
    "portrait": Layout(name="portrait", play_res_x=1080, play_res_y=1920, font_size=64, safe_area=0.84,
                       max_lines=2, margin_v=420, max_cps=15.0),
    "square": Layout(name="square", play_res_x=1080, play_res_y=1080, font_size=60, max_lines=2, margin_v=80),
}


def get_layout(name_or_res: str = "landscape", font_size: Optional[int] = None,
               max_units: Optional[int] = None, max_lines: Optional[int] = None) -> Layout:
    """``name_or_res``: landscape | portrait | square | WIDTHxHEIGHT."""
    if "x" in name_or_res and name_or_res.replace("x", "").isdigit():
        w, h = (int(v) for v in name_or_res.split("x"))
        base = LAYOUTS["portrait" if h > w * 1.1 else "landscape" if w > h * 1.1 else "square"]
        scale = (w if h > w else h) / (base.play_res_x if h > w else base.play_res_y)
        lay = replace(base, play_res_x=w, play_res_y=h, font_size=int(round(base.font_size * scale)),
                      margin_v=int(round(base.margin_v * scale)))
    else:
        if name_or_res not in LAYOUTS:
            raise ValueError(f"unknown layout {name_or_res!r}; use {sorted(LAYOUTS)} or WxH")
        lay = replace(LAYOUTS[name_or_res])
    if font_size:
        lay.font_size = font_size
    if max_units:
        lay.max_units_override = max_units
    if max_lines:
        lay.max_lines = max_lines
    return lay
