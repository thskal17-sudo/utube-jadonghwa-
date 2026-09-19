"""폴더 감시 자동 실행 테스트."""

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from shorts_pipeline.config import ShortsConfig
from shorts_pipeline.watcher import (
    DESCRIPTION_FILENAME,
    STATE_FILENAME,
    Fingerprint,
    FolderWatcher,
    LockError,
    SingleInstanceLock,
    WatchState,
    is_quiet,
    list_photos,
    read_description,
)


def make_photo(path: Path, seed: int = 0, w: int = 400, h: int = 300) -> Path:
    rng = np.random.default_rng(seed)
    cv2.imwrite(str(path), rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    return path


def age_files(folder: Path, seconds: float) -> None:
    """파일 수정 시각을 과거로 돌려 '복사가 끝난' 상태를 만든다."""
    past = time.time() - seconds
    for p in folder.iterdir():
        os.utime(p, (past, past))


@pytest.fixture
def job(tmp_path):
    watch = tmp_path / "watch"
    folder = watch / "과학실험"
    folder.mkdir(parents=True)
    for i in range(3):
        make_photo(folder / f"{i}.jpg", i)
    age_files(folder, 600)
    return watch, folder


# ---------- 사진 수집 ----------
def test_list_photos_skips_non_images_and_hidden(tmp_path):
    d = tmp_path / "f"
    d.mkdir()
    make_photo(d / "a.jpg")
    (d / "notes.txt").write_text("x", encoding="utf-8")
    (d / ".hidden.jpg").write_text("x", encoding="utf-8")
    assert [p.name for p in list_photos(d)] == ["a.jpg"]


# ---------- 활동 설명 ----------
def test_description_file_wins(tmp_path):
    d = tmp_path / "운동회"
    d.mkdir()
    (d / DESCRIPTION_FILENAME).write_text("가을 운동회\n이어달리기", encoding="utf-8")
    assert read_description(d) == "가을 운동회 이어달리기"


def test_folder_name_is_the_fallback(tmp_path):
    d = tmp_path / "미술_시간"
    d.mkdir()
    assert read_description(d) == "미술 시간"


def test_empty_description_file_falls_back(tmp_path):
    d = tmp_path / "과학실험"
    d.mkdir()
    (d / DESCRIPTION_FILENAME).write_text("   \n", encoding="utf-8")
    assert read_description(d) == "과학실험"


# ---------- 안정 판정 (디바운스) ----------
def test_recent_files_are_not_quiet():
    fp = Fingerprint(count=3, total_size=100, latest_mtime=time.time())
    assert not is_quiet(fp, quiet_seconds=180)


def test_old_files_are_quiet():
    fp = Fingerprint(count=3, total_size=100, latest_mtime=time.time() - 600)
    assert is_quiet(fp, quiet_seconds=180)


def test_future_mtime_is_never_quiet():
    """시계가 어긋난 네트워크 드라이브에서 미래 시각이 찍히는 일이 있다."""
    fp = Fingerprint(count=1, total_size=10, latest_mtime=time.time() + 3600)
    assert not is_quiet(fp, quiet_seconds=180)


def test_fingerprint_detects_added_photo(job):
    _, folder = job
    before = Fingerprint.of(list_photos(folder))
    make_photo(folder / "extra.jpg", 99)
    after = Fingerprint.of(list_photos(folder))
    assert not after.matches({"count": before.count, "total_size": before.total_size})


# ---------- 상태 저장 ----------
def test_state_roundtrip(tmp_path):
    path = tmp_path / STATE_FILENAME
    state = WatchState()
    state.record("과학실험", "done", Fingerprint(3, 100, 1.0), output="a.mp4")
    state.save(path)
    loaded = WatchState.load(path)
    assert loaded.jobs["과학실험"]["status"] == "done"
    assert loaded.jobs["과학실험"]["output"] == "a.mp4"


def test_corrupt_state_file_does_not_crash(tmp_path):
    path = tmp_path / STATE_FILENAME
    path.write_text("{ 깨진 json", encoding="utf-8")
    assert WatchState.load(path).jobs == {}


def test_state_save_is_atomic(tmp_path):
    """임시 파일을 남기지 않는다. 남으면 다음 읽기가 헷갈린다."""
    path = tmp_path / STATE_FILENAME
    state = WatchState()
    state.record("a", "done", Fingerprint())
    state.save(path)
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(path.read_text(encoding="utf-8"))["jobs"]["a"]["status"] == "done"


# ---------- 잠금 ----------
def test_lock_blocks_second_holder(tmp_path):
    path = tmp_path / "w.lock"
    with SingleInstanceLock(path):
        with pytest.raises(LockError):
            with SingleInstanceLock(path):
                pass


def test_lock_is_released_on_exit(tmp_path):
    path = tmp_path / "w.lock"
    with SingleInstanceLock(path):
        assert path.exists()
    assert not path.exists()


def test_lock_released_even_on_exception(tmp_path):
    path = tmp_path / "w.lock"
    with pytest.raises(RuntimeError):
        with SingleInstanceLock(path):
            raise RuntimeError("작업 중 오류")
    assert not path.exists()


def test_stale_lock_from_dead_process_is_reclaimed(tmp_path):
    path = tmp_path / "w.lock"
    path.write_text(json.dumps({"pid": 999_999, "started": "x"}), encoding="utf-8")
    with SingleInstanceLock(path):
        assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_unreadable_lock_is_reclaimed(tmp_path):
    path = tmp_path / "w.lock"
    path.write_text("쓰레기", encoding="utf-8")
    with SingleInstanceLock(path):
        assert path.exists()


# ---------- 처리 판단 ----------
def _watcher(watch, tmp_path, **kw):
    cfg = ShortsConfig(width=180, height=320, fps=6, duration=3.0,
                       preset="ultrafast", video_bitrate="300k")
    return FolderWatcher(watch, tmp_path / "out", config=cfg, **{"quiet_seconds": 1.0, **kw})


def test_ready_folder_is_selected(job, tmp_path):
    watch, folder = job
    should, reason, fp = _watcher(watch, tmp_path).evaluate(folder)
    assert should, reason
    assert fp.count == 3


def test_folder_still_being_copied_is_skipped(job, tmp_path):
    watch, folder = job
    make_photo(folder / "new.jpg", 42)  # 방금 만든 파일
    should, reason, _ = _watcher(watch, tmp_path, quiet_seconds=600).evaluate(folder)
    assert not should
    assert "들어오는 중" in reason


def test_too_few_photos_is_skipped(tmp_path):
    watch = tmp_path / "watch"
    folder = watch / "한장뿐"
    folder.mkdir(parents=True)
    make_photo(folder / "a.jpg")
    age_files(folder, 600)
    should, reason, _ = _watcher(watch, tmp_path).evaluate(folder)
    assert not should
    assert "최소" in reason


def test_already_done_folder_is_skipped(job, tmp_path):
    watch, folder = job
    w = _watcher(watch, tmp_path)
    fp = Fingerprint.of(list_photos(folder))
    w.state.record(folder.name, "done", fp)
    should, reason, _ = w.evaluate(folder)
    assert not should
    assert "이미 처리" in reason


def test_failed_folder_is_not_retried_automatically(job, tmp_path):
    """같은 사진으로 다시 돌려도 같은 결과일 가능성이 크다."""
    watch, folder = job
    w = _watcher(watch, tmp_path)
    w.state.record(folder.name, "failed", Fingerprint(count=99, total_size=99))
    should, reason, _ = w.evaluate(folder)
    assert not should
    assert "실패" in reason


def test_reprocess_overrides_done_state(job, tmp_path):
    watch, folder = job
    w = _watcher(watch, tmp_path, reprocess=True)
    w.state.record(folder.name, "done", Fingerprint.of(list_photos(folder)))
    should, _, _ = w.evaluate(folder)
    assert should


def test_missing_watch_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _watcher(tmp_path / "nope", tmp_path).job_folders()


def test_hidden_folders_are_ignored(tmp_path):
    watch = tmp_path / "watch"
    (watch / ".git").mkdir(parents=True)
    (watch / "진짜").mkdir()
    assert [p.name for p in _watcher(watch, tmp_path).job_folders()] == ["진짜"]


# ---------- 실제 처리 ----------
@pytest.mark.slow
def test_scan_once_renders_and_records(job, tmp_path):
    watch, folder = job
    w = _watcher(watch, tmp_path)
    results = w.scan_once()
    done = [r for r in results if r.status == "done"]
    assert len(done) == 1
    assert Path(done[0].output).exists()
    assert w.state.jobs[folder.name]["status"] == "done"
    assert (tmp_path / "out" / STATE_FILENAME).exists()


@pytest.mark.slow
def test_second_scan_does_nothing(job, tmp_path):
    watch, _ = job
    w = _watcher(watch, tmp_path)
    w.scan_once()
    again = w.scan_once()
    assert all(r.status == "skipped" for r in again)


@pytest.mark.slow
def test_failure_is_recorded_and_does_not_raise(tmp_path):
    watch = tmp_path / "watch"
    folder = watch / "깨진사진"
    folder.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (folder / name).write_bytes(os.urandom(300))  # 이미지가 아니다
    age_files(folder, 600)
    w = _watcher(watch, tmp_path)
    results = w.scan_once()
    assert [r.status for r in results] == ["failed"]
    assert w.state.jobs["깨진사진"]["status"] == "failed"
    assert "error" in w.state.jobs["깨진사진"]


@pytest.mark.slow
def test_description_file_reaches_the_video(job, tmp_path):
    watch, folder = job
    (folder / DESCRIPTION_FILENAME).write_text("화산 폭발 실험", encoding="utf-8")
    age_files(folder, 600)
    w = _watcher(watch, tmp_path)
    w.scan_once()
    captions = json.loads(
        (tmp_path / "out" / f"{folder.name}_work" / "captions.json").read_text(encoding="utf-8")
    )
    assert captions["captions"][0] == "화산 폭발 실험"
