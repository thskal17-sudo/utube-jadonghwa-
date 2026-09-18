import numpy as np
import pytest
from PIL import ImageFont

from shorts_pipeline.fonts import find_korean_font, font_supports_hangul
from shorts_pipeline.subtitles import render_text_block, wrap_text

FONT = find_korean_font()


def test_found_font_supports_hangul():
    assert font_supports_hangul(FONT)


def test_wrap_text_respects_max_width():
    font = ImageFont.truetype(str(FONT), 40)
    lines = wrap_text("오늘 우리 반은 과학실에서 화산 폭발 실험을 했습니다", font, 300)
    assert len(lines) > 1
    assert all(font.getlength(line) <= 300 for line in lines)


def test_wrap_text_breaks_unbreakable_word():
    font = ImageFont.truetype(str(FONT), 40)
    lines = wrap_text("가" * 60, font, 200)
    assert len(lines) > 1
    assert all(font.getlength(line) <= 200 for line in lines)


def test_wrap_text_keeps_explicit_newlines():
    font = ImageFont.truetype(str(FONT), 30)
    assert len(wrap_text("첫 줄\n둘째 줄", font, 2000)) == 2


def test_render_text_block_is_rgba_and_within_width():
    img = render_text_block("과학 실험 재밌다!", FONT, 48, max_width=800)
    assert img.ndim == 3 and img.shape[2] == 4
    assert img.shape[1] <= 800
    assert img[..., 3].max() == 255  # 불투명 픽셀 존재


def test_render_text_block_draws_visible_glyphs():
    blank = render_text_block(" ", FONT, 48, max_width=600, bg=None)
    text = render_text_block("한글 자막", FONT, 48, max_width=600, bg=None)
    assert text[..., 3].sum() > blank[..., 3].sum() * 3


def test_render_long_text_wraps_taller():
    short = render_text_block("짧게", FONT, 44, max_width=600)
    long = render_text_block("아주 긴 자막 문장을 여러 줄로 감싸야 하는 경우입니다 " * 2, FONT, 44, max_width=600)
    assert long.shape[0] > short.shape[0]
    assert long.shape[1] <= 600
