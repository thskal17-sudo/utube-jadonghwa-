import numpy as np
import pytest

from shorts_pipeline.bgm import SAMPLE_RATE, STYLES, synthesize_bgm, write_bgm


@pytest.mark.parametrize("style", sorted(STYLES))
def test_bgm_length_and_range(style):
    audio = synthesize_bgm(4.0, style=style, seed=1)
    assert len(audio) == int(4.0 * SAMPLE_RATE)
    assert audio.dtype == np.float32
    assert np.max(np.abs(audio)) <= 1.0
    assert np.max(np.abs(audio)) > 0.1  # 무음이 아니어야 한다


def test_bgm_is_deterministic_for_seed():
    a = synthesize_bgm(3.0, style="bright", seed=42)
    b = synthesize_bgm(3.0, style="bright", seed=42)
    assert np.array_equal(a, b)


def test_bgm_seed_changes_output():
    a = synthesize_bgm(3.0, style="energetic", seed=1)
    b = synthesize_bgm(3.0, style="energetic", seed=2)
    assert not np.array_equal(a, b)


def test_bgm_fades_at_both_ends():
    audio = synthesize_bgm(5.0, style="bright", seed=0)
    assert abs(audio[0]) < 0.02
    assert abs(audio[-1]) < 0.02


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        synthesize_bgm(2.0, style="doesnotexist")


def test_write_bgm_creates_wav(tmp_path):
    path = write_bgm(tmp_path / "sub" / "bgm.wav", 2.0, style="calm", seed=0)
    assert path.exists() and path.stat().st_size > 1000
