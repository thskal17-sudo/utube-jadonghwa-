import numpy as np
import pytest

from shorts_pipeline.kenburns import KenBurnsScene, KenBurnsParams, smoothstep
from shorts_pipeline.render import compute_scene_timing


def test_scene_timing_total_duration():
    for n in range(1, 9):
        timings = compute_scene_timing(n, 18.0, 0.6)
        assert len(timings) == n
        last_start, last_dur = timings[-1]
        assert last_start + last_dur == pytest.approx(18.0)


def test_scene_timing_scenes_overlap_by_transition():
    timings = compute_scene_timing(4, 18.0, 0.6)
    for (s0, d0), (s1, _) in zip(timings, timings[1:]):
        assert s0 + d0 - s1 == pytest.approx(0.6)


def test_scene_timing_single_scene():
    assert compute_scene_timing(1, 15.0, 0.6) == [(0.0, 15.0)]


def test_scene_timing_handles_many_short_scenes():
    # 전환 시간이 장면 길이보다 길어지는 경우에도 총 길이는 유지된다
    timings = compute_scene_timing(12, 15.0, 2.0)
    assert timings[-1][0] + timings[-1][1] == pytest.approx(15.0)
    assert all(d > 0 for _, d in timings)


def test_smoothstep_bounds():
    assert smoothstep(-1) == 0.0
    assert smoothstep(0) == 0.0
    assert smoothstep(1) == 1.0
    assert smoothstep(2) == 1.0
    assert 0.0 < smoothstep(0.5) < 1.0


@pytest.mark.parametrize("shape", [(1200, 1600, 3), (1600, 1200, 3), (900, 900, 3)])
def test_kenburns_frame_shape_and_range(shape):
    img = np.random.default_rng(0).integers(0, 255, shape, dtype=np.uint8)
    scene = KenBurnsScene(img, 1080, 1920, 3.0, KenBurnsParams(1.0, 1.18, (-0.5, 0), (0.5, 0)))
    for t in (0.0, 1.5, 3.0):
        frame = scene.frame(t)
        assert frame.shape == (1920, 1080, 3)
        assert frame.dtype == np.uint8


def test_kenburns_frames_actually_move():
    img = np.random.default_rng(1).integers(0, 255, (1200, 1600, 3), dtype=np.uint8)
    scene = KenBurnsScene(img, 1080, 1920, 3.0, KenBurnsParams(1.0, 1.18, (-0.6, 0), (0.6, 0)))
    first, last = scene.frame(0.0), scene.frame(3.0)
    assert not np.array_equal(first, last)
