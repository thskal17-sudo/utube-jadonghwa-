"""좌표 읽기용 격자 생성 도구 테스트."""

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from make_grid import draw_grid  # noqa: E402


def test_draw_grid_keeps_aspect_and_scales():
    img = np.full((600, 800, 3), 128, dtype=np.uint8)
    out = draw_grid(img, step=100, scale=0.5)
    assert out.shape[0] == 300 and out.shape[1] == 400


def test_draw_grid_draws_visible_lines():
    img = np.full((400, 400, 3), 0, dtype=np.uint8)
    out = draw_grid(img, step=100, scale=1.0)
    # 격자는 노란색(0,255,255) → 초록/빨강 채널에 값이 생긴다
    assert out[..., 1].max() > 200
    assert out.sum() > img.sum()


@pytest.mark.parametrize("step", [50, 100, 250])
def test_draw_grid_accepts_various_steps(step):
    img = np.full((500, 500, 3), 90, dtype=np.uint8)
    assert draw_grid(img, step=step, scale=1.0).shape == (500, 500, 3)


def test_cli_creates_grid_image(tmp_path):
    src = tmp_path / "photo.jpg"
    cv2.imwrite(str(src), np.full((600, 900, 3), 170, dtype=np.uint8))
    dst = tmp_path / "out.jpg"
    proc = subprocess.run(
        [sys.executable, "scripts/make_grid.py", str(src), "-o", str(dst), "--step", "150"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-1000:]
    assert dst.exists()
    assert cv2.imread(str(dst)) is not None


def test_cli_missing_file_exits_nonzero(tmp_path):
    proc = subprocess.run(
        [sys.executable, "scripts/make_grid.py", str(tmp_path / "nope.jpg")],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode != 0
