"""파이프라인 전 구간 통합 테스트 (합성 사진 → 실제 mp4)."""

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from shorts_pipeline import ShortsConfig, run_pipeline
from shorts_pipeline.captions import CaptionPlan
from shorts_pipeline.pipeline import collect_images

FFMPEG = None
try:
    import imageio_ffmpeg

    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    pass


def make_photo(path: Path, w: int, h: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 210, dtype=np.uint8)
    for _ in range(8):
        x, y = int(rng.integers(0, w - 50)), int(rng.integers(0, h - 50))
        cv2.rectangle(img, (x, y), (x + 60, y + 50), tuple(int(c) for c in rng.integers(60, 240, 3)), -1)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


@pytest.fixture
def photo_dir(tmp_path):
    d = tmp_path / "photos"
    d.mkdir()
    for i in range(3):
        make_photo(d / f"p{i + 1}.jpg", 800 if i % 2 == 0 else 600, 600 if i % 2 == 0 else 800, i)
    return d


def test_collect_images_sorts_naturally(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    for name in ["img10.jpg", "img2.jpg", "img1.jpg"]:
        make_photo(d / name, 200, 200, 0)
    names = [p.name for p in collect_images(d, 10)]
    assert names == ["img1.jpg", "img2.jpg", "img10.jpg"]


def test_collect_images_subsamples(tmp_path):
    d = tmp_path / "p"
    d.mkdir()
    for i in range(20):
        make_photo(d / f"i{i:02d}.jpg", 200, 200, i)
    assert len(collect_images(d, 5)) == 5


def test_collect_images_empty_dir_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        collect_images(d, 5)


@pytest.mark.slow
def test_end_to_end_produces_playable_shorts(photo_dir, tmp_path):
    out = tmp_path / "out.mp4"
    cfg = ShortsConfig(width=270, height=480, fps=8, duration=4.0, preset="ultrafast", video_bitrate="500k")
    report = run_pipeline(photo_dir, "테스트 활동 설명", out, cfg, work_dir=tmp_path / "work")

    assert out.exists() and out.stat().st_size > 5000
    assert len(report.photos_used) == 3
    assert report.resolution == "270x480"
    assert Path(report.review_sheet).exists()
    assert Path(report.bgm).exists()
    assert json.loads((tmp_path / "work" / "captions.json").read_text(encoding="utf-8"))["title"]

    cap = cv2.VideoCapture(str(out))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 270
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 480
        ok, frame = cap.read()
        assert ok and frame.shape == (480, 270, 3)
    finally:
        cap.release()


@pytest.mark.slow
@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg 없음")
def test_output_has_audio_and_yuv420p(photo_dir, tmp_path):
    out = tmp_path / "a.mp4"
    cfg = ShortsConfig(width=270, height=480, fps=8, duration=4.0, preset="ultrafast", video_bitrate="500k")
    run_pipeline(photo_dir, "소리 확인", out, cfg, work_dir=tmp_path / "w")
    info = subprocess.run([FFMPEG, "-hide_banner", "-i", str(out)], capture_output=True, text=True).stderr
    assert "Audio: aac" in info
    assert "yuv420p" in info


@pytest.mark.slow
def test_custom_caption_plan_is_used(photo_dir, tmp_path):
    out = tmp_path / "c.mp4"
    plan = CaptionPlan("커스텀 제목", ["첫 장면", "둘째 장면"], ["#태그"])
    cfg = ShortsConfig(width=270, height=480, fps=8, duration=4.0, preset="ultrafast", video_bitrate="500k")
    report = run_pipeline(photo_dir, "무시됨", out, cfg, work_dir=tmp_path / "w", caption_plan=plan)
    assert report.caption_plan["title"] == "커스텀 제목"
    assert report.caption_plan["captions"][0] == "첫 장면"
    assert len(report.caption_plan["captions"]) == 3  # 장면 수에 맞춰 순환


@pytest.mark.slow
def test_duration_outside_shorts_range_warns(photo_dir, tmp_path):
    cfg = ShortsConfig(width=270, height=480, fps=8, duration=4.0, preset="ultrafast", video_bitrate="500k")
    report = run_pipeline(photo_dir, "짧은 영상", tmp_path / "s.mp4", cfg, work_dir=tmp_path / "w")
    assert any("권장 범위" in w for w in report.warnings)


@pytest.mark.slow
def test_no_blur_flag_warns(photo_dir, tmp_path):
    cfg = ShortsConfig(width=270, height=480, fps=8, duration=4.0, blur=False, preset="ultrafast", video_bitrate="500k")
    report = run_pipeline(photo_dir, "블러 없음", tmp_path / "n.mp4", cfg, work_dir=tmp_path / "w")
    assert any("블러를 건너뛰" in w for w in report.warnings)
    assert report.review_sheet is None


@pytest.mark.slow
def test_cli_runs(photo_dir, tmp_path):
    out = tmp_path / "cli.mp4"
    proc = subprocess.run(
        [sys.executable, "make_shorts.py", str(photo_dir), "-d", "명령줄 테스트",
         "-o", str(out), "--preview", "--duration", "4", "--work-dir", str(tmp_path / "w")],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert out.exists()


def test_cli_missing_folder_exits_nonzero(tmp_path):
    proc = subprocess.run(
        [sys.executable, "make_shorts.py", str(tmp_path / "nope"), "-d", "x"],
        capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent,
    )
    assert proc.returncode != 0
