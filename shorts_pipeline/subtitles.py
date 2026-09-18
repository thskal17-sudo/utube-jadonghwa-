"""자막/제목 텍스트를 이미지(RGBA)로 렌더링 (Pillow)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

RGBA = Tuple[int, int, int, int]


def _wrap_line(line: str, font: ImageFont.FreeTypeFont, max_width: float) -> List[str]:
    """공백 단위로 줄바꿈하고, 한 단어가 너무 길면 글자 단위로 자른다."""
    words = line.split(" ")
    lines: List[str] = []
    cur = ""
    for word in words:
        trial = (cur + " " + word).strip()
        if font.getlength(trial) <= max_width:
            cur = trial
            continue
        if cur:
            lines.append(cur)
        cur = word
        while font.getlength(cur) > max_width and len(cur) > 1:
            cut = len(cur)
            while cut > 1 and font.getlength(cur[:cut]) > max_width:
                cut -= 1
            lines.append(cur[:cut])
            cur = cur[cut:]
    if cur:
        lines.append(cur)
    return lines or [""]


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: float) -> List[str]:
    lines: List[str] = []
    for para in text.split("\n"):
        lines.extend(_wrap_line(para, font, max_width))
    return lines


def render_text_block(
    text: str,
    font_path: Path,
    font_size: int,
    max_width: int,
    color: RGBA = (255, 255, 255, 255),
    stroke_color: RGBA = (0, 0, 0, 255),
    stroke_width: Optional[int] = None,
    bg: Optional[RGBA] = (0, 0, 0, 150),
    padding: Tuple[int, int] = (32, 20),
    radius: int = 26,
    line_spacing: float = 1.25,
) -> np.ndarray:
    """텍스트를 둥근 반투명 상자 위에 그려 RGBA numpy 배열로 돌려준다."""
    font = ImageFont.truetype(str(font_path), font_size)
    if stroke_width is None:
        stroke_width = max(2, int(font_size * 0.07))
    inner_w = max_width - 2 * padding[0] - 2 * stroke_width
    lines = wrap_text(text.strip(), font, inner_w)

    line_h = int(font_size * line_spacing)
    text_w = max(font.getlength(line) for line in lines)
    ascent, descent = font.getmetrics()
    text_h = line_h * (len(lines) - 1) + ascent + descent

    W = int(text_w + 2 * padding[0] + 2 * stroke_width)
    H = int(text_h + 2 * padding[1] + 2 * stroke_width)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if bg and bg[3] > 0:
        draw.rounded_rectangle([0, 0, W - 1, H - 1], radius=radius, fill=bg)

    y = padding[1] + stroke_width
    for line in lines:
        x = (W - font.getlength(line)) / 2
        draw.text(
            (x, y), line, font=font, fill=color,
            stroke_width=stroke_width, stroke_fill=stroke_color, anchor="la",
        )
        y += line_h
    return np.asarray(img)
