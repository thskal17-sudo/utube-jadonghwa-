"""watch_shorts.py 명령줄 동작 테스트."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


def make_photo(path: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (300, 400, 3), dtype=np.uint8))


def ready_job(tmp_path, name="과학실험", n=3):
    watch = tmp_path / "watch"
    folder = watch / name
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        make_photo(folder / f"{i}.jpg", i)
    past = time.time() - 600
    for p in folder.iterdir():
        os.utime(p, (past, past))
    return watch, folder


def run_cli(*args, cwd=ROOT):
    return subprocess.run(
        [sys.executable, "watch_shorts.py", *[str(a) for a in args]],
        capture_output=True, text=True, cwd=cwd,
    )


def test_dry_run_lists_without_rendering(tmp_path):
    watch, _ = ready_job(tmp_path)
    out = tmp_path / "out"
    proc = run_cli(watch, "-o", out, "--dry-run")
    assert proc.returncode == 0, proc.stderr[-800:]
    assert "과학실험" in proc.stdout
    assert "처리함" in proc.stdout
    assert not list(out.glob("*.mp4")) if out.exists() else True


def test_dry_run_reports_waiting_folder(tmp_path):
    watch, folder = ready_job(tmp_path)
    make_photo(folder / "fresh.jpg", 9)  # 방금 복사된 파일
    proc = run_cli(watch, "-o", tmp_path / "out", "--dry-run", "--quiet-seconds", "600")
    assert proc.returncode == 0
    assert "건너뜀" in proc.stdout


def test_missing_watch_dir_exits_nonzero(tmp_path):
    proc = run_cli(tmp_path / "nope", "-o", tmp_path / "out", "--dry-run")
    assert proc.returncode != 0
    assert "오류" in proc.stderr


def test_lock_blocks_concurrent_run(tmp_path):
    watch, _ = ready_job(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    lock = out / ".watch.lock"
    lock.write_text(json.dumps({"pid": os.getpid(), "started": "x"}), encoding="utf-8")
    proc = run_cli(watch, "-o", out, "--once", "--preview")
    assert proc.returncode != 0
    assert "실행 중" in proc.stderr


@pytest.mark.slow
def test_once_renders_and_reports(tmp_path):
    watch, _ = ready_job(tmp_path)
    out = tmp_path / "out"
    proc = run_cli(watch, "-o", out, "--once", "--preview", "--duration", "4")
    assert proc.returncode == 0, proc.stderr[-1500:]
    assert (out / "과학실험.mp4").exists()
    assert "검수 필요" in proc.stdout
    assert (out / ".watch_state.json").exists()
    assert not (out / ".watch.lock").exists()  # 잠금은 해제돼야 한다


@pytest.mark.slow
def test_once_is_idempotent(tmp_path):
    watch, _ = ready_job(tmp_path)
    out = tmp_path / "out"
    run_cli(watch, "-o", out, "--once", "--preview", "--duration", "4")
    mtime = (out / "과학실험.mp4").stat().st_mtime
    proc = run_cli(watch, "-o", out, "--once", "--preview", "--duration", "4")
    assert proc.returncode == 0
    assert "처리 0건" in proc.stdout
    assert (out / "과학실험.mp4").stat().st_mtime == mtime


@pytest.mark.slow
def test_failure_exits_nonzero(tmp_path):
    watch = tmp_path / "watch"
    folder = watch / "깨진사진"
    folder.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (folder / name).write_bytes(os.urandom(300))
    past = time.time() - 600
    for p in folder.iterdir():
        os.utime(p, (past, past))
    proc = run_cli(watch, "-o", tmp_path / "out", "--once", "--preview", "--duration", "4")
    assert proc.returncode == 1
    assert "실패" in proc.stdout
