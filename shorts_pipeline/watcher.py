"""Phase 4: 폴더 감시 자동 실행.

지정한 폴더 아래의 하위 폴더 하나를 영상 한 편으로 보고, 사진이 다 들어온 것이
확인되면 파이프라인을 돌린다.

    watch/
    ├── 과학실험/            ← 영상 한 편
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── description.txt  (선택: 활동 설명. 없으면 폴더명을 쓴다)
    └── 운동회/
        └── ...

왜 watchdog 대신 폴링인가
원래 설계는 watchdog(inotify/FSEvents)이었다. 실제로 쓸 환경을 생각하니 폴링이
맞다고 판단해 바꿨다.

1. 학교에서는 사진이 네트워크 드라이브나 클라우드 동기화 폴더(OneDrive, 구글
   드라이브)에 올라온다. 이런 곳에서는 파일시스템 이벤트가 아예 안 오거나
   뒤늦게 온다. 이벤트를 놓치면 영상이 영영 안 만들어진다.
2. 이벤트를 받아도 "복사가 끝났는지"는 알 수 없어서 결국 안정될 때까지 폴링해야
   한다. 사진 여덟 장을 복사하는 중에 실행되면 일부만 들어간 영상이 나온다.
3. 의존성이 하나 줄어든다.

폴링 주기가 분 단위여도 문제가 없다. 사람이 사진을 올리고 바로 결과를 기다리는
작업이 아니기 때문이다.

안정 판정
가장 최근에 수정된 파일이 quiet_seconds 보다 오래됐으면 '다 들어왔다'고 본다.
이 판정은 상태를 들고 있지 않아도 되므로, 상주 실행과 cron 실행이 똑같이 동작한다.

자동 게시는 하지 않는다
결과는 output/ 에 두고 끝낸다. 얼굴 블러는 완벽하지 않고 자막은 생성형이라,
사람이 보지 않은 영상이 채널에 올라가면 안 된다.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import IMAGE_EXTENSIONS, ShortsConfig
from .pipeline import run_pipeline

log = logging.getLogger(__name__)

STATE_FILENAME = ".watch_state.json"
LOCK_FILENAME = ".watch.lock"
DESCRIPTION_FILENAME = "description.txt"
DEFAULT_QUIET_SECONDS = 180.0   # 사진 복사가 끝났다고 볼 정지 시간
DEFAULT_POLL_SECONDS = 60.0
MIN_PHOTOS = 2


@dataclass
class Fingerprint:
    """폴더 내용을 요약한 값. 이게 바뀌면 사진이 달라진 것이다."""

    count: int = 0
    total_size: int = 0
    latest_mtime: float = 0.0

    @classmethod
    def of(cls, photos: List[Path]) -> "Fingerprint":
        return cls(
            count=len(photos),
            total_size=sum(p.stat().st_size for p in photos),
            latest_mtime=max((p.stat().st_mtime for p in photos), default=0.0),
        )

    def matches(self, other: Optional[dict]) -> bool:
        if not other:
            return False
        return (
            other.get("count") == self.count
            and other.get("total_size") == self.total_size
        )


@dataclass
class JobResult:
    name: str
    status: str            # done | failed | skipped | waiting
    message: str = ""
    output: Optional[str] = None


@dataclass
class WatchState:
    """어떤 폴더를 처리했는지 기록한다. 재시작해도 다시 만들지 않기 위해서다."""

    jobs: Dict[str, dict] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "WatchState":
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(jobs=dict(data.get("jobs", {})))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("[watch] 상태 파일을 읽지 못해 새로 시작합니다: %s", e)
            return cls()

    def save(self, path: Path) -> None:
        """임시 파일에 쓰고 바꿔치기한다. 도중에 죽어도 상태가 깨지지 않는다."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps({"jobs": self.jobs}, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def record(self, name: str, status: str, fingerprint: Fingerprint, **extra) -> None:
        self.jobs[name] = {
            "status": status,
            "fingerprint": asdict(fingerprint),
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **extra,
        }


class LockError(RuntimeError):
    pass


class SingleInstanceLock:
    """동시 실행을 막는다. 렌더링이 CPU 를 오래 쓰므로 두 개가 겹치면 곤란하다."""

    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def _stale(self) -> bool:
        try:
            pid = int(json.loads(self.path.read_text(encoding="utf-8"))["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return True   # 읽을 수 없는 잠금은 버린다
        try:
            os.kill(pid, 0)   # 신호 0 = 존재 확인만
        except ProcessLookupError:
            return True       # 그 프로세스는 죽었다
        except PermissionError:
            return False      # 다른 사용자의 살아 있는 프로세스
        return False
        # 주의: pid 가 자기 자신이어도 '죽었다'고 보지 않는다. 그렇게 하면
        # 같은 프로세스가 잠금을 두 번 잡을 수 있다. 이미 잡고 있다면 막는 게 맞다.

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self._stale():
            log.warning("[watch] 남아 있던 잠금 파일을 정리합니다: %s", self.path)
            self.path.unlink(missing_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise LockError(
                f"이미 실행 중입니다(잠금: {self.path}). 아니라면 그 파일을 지우세요."
            ) from e
        with os.fdopen(fd, "w") as f:
            json.dump({"pid": os.getpid(), "started": datetime.now(timezone.utc).isoformat()}, f)
        self.acquired = True
        return self

    def __exit__(self, *exc):
        if self.acquired:
            self.path.unlink(missing_ok=True)
        return False


def list_photos(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.startswith(".")
    )


def read_description(folder: Path) -> str:
    """description.txt 가 있으면 그 내용을, 없으면 폴더명을 활동 설명으로 쓴다."""
    desc_file = folder / DESCRIPTION_FILENAME
    if desc_file.exists():
        text = " ".join(desc_file.read_text(encoding="utf-8").split())
        if text:
            return text
    return folder.name.replace("_", " ").strip()


def is_quiet(fingerprint: Fingerprint, quiet_seconds: float, now: Optional[float] = None) -> bool:
    """마지막 수정 이후 충분히 조용했는지.

    사진을 여러 장 복사하는 중에 실행하면 일부만 들어간 영상이 나온다.
    미래 시각을 가진 파일(시계 어긋남, 네트워크 드라이브)은 조용하지 않다고 본다.
    """
    now = time.time() if now is None else now
    age = now - fingerprint.latest_mtime
    if age < 0:
        return False   # 미래 시각 = 시계가 어긋났다. 안전하게 더 기다린다.
    return age >= quiet_seconds


class FolderWatcher:
    """감시 폴더를 한 번 훑거나(scan_once) 계속 돈다(run_forever)."""

    def __init__(
        self,
        watch_dir: Path,
        output_dir: Path,
        config: Optional[ShortsConfig] = None,
        quiet_seconds: float = DEFAULT_QUIET_SECONDS,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        min_photos: int = MIN_PHOTOS,
        reprocess: bool = False,
    ):
        self.watch_dir = Path(watch_dir)
        self.output_dir = Path(output_dir)
        self.config = config or ShortsConfig()
        self.quiet_seconds = quiet_seconds
        self.poll_seconds = poll_seconds
        self.min_photos = min_photos
        self.reprocess = reprocess
        self.state_path = self.output_dir / STATE_FILENAME
        self.state = WatchState.load(self.state_path)
        self._stop = False

    # ------------------------------------------------------------------
    def job_folders(self) -> List[Path]:
        if not self.watch_dir.is_dir():
            raise FileNotFoundError(f"감시할 폴더가 없습니다: {self.watch_dir}")
        return sorted(
            p for p in self.watch_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def evaluate(self, folder: Path) -> tuple:
        """(처리할지, 사유, 지문) 을 돌려준다."""
        photos = list_photos(folder)
        fp = Fingerprint.of(photos)
        if len(photos) < self.min_photos:
            return False, f"사진이 {len(photos)}장뿐입니다(최소 {self.min_photos}장)", fp

        record = self.state.jobs.get(folder.name)
        if record and not self.reprocess:
            if fp.matches(record.get("fingerprint")):
                return False, f"이미 처리했습니다({record.get('status')})", fp
            if record.get("status") == "failed":
                return False, "이전에 실패했습니다. --reprocess 로 다시 시도하세요", fp
            return False, "처리 뒤 사진이 바뀌었습니다. --reprocess 로 다시 만드세요", fp

        if not is_quiet(fp, self.quiet_seconds):
            waited = max(0, time.time() - fp.latest_mtime)
            return False, f"아직 파일이 들어오는 중입니다({waited:.0f}/{self.quiet_seconds:.0f}초)", fp
        return True, "", fp

    def process(self, folder: Path, fingerprint: Fingerprint) -> JobResult:
        description = read_description(folder)
        output = self.output_dir / f"{folder.name}.mp4"
        log.info("[watch] 처리 시작: %s (설명: %s)", folder.name, description)
        try:
            report = run_pipeline(
                folder, description, output, self.config,
                work_dir=self.output_dir / f"{folder.name}_work",
            )
        except Exception as e:
            # 같은 사진으로 다시 돌려도 같은 결과일 가능성이 크다. 재시도하지 않는다.
            log.exception("[watch] 실패: %s", folder.name)
            self.state.record(folder.name, "failed", fingerprint, error=f"{type(e).__name__}: {e}")
            self.state.save(self.state_path)
            return JobResult(folder.name, "failed", str(e))

        self.state.record(
            folder.name, "done", fingerprint,
            output=str(output), warnings=report.warnings,
        )
        self.state.save(self.state_path)
        log.info("[watch] 완료: %s → %s (검수 필요)", folder.name, output)
        return JobResult(folder.name, "done", "검수 대기", str(output))

    def scan_once(self) -> List[JobResult]:
        results: List[JobResult] = []
        for folder in self.job_folders():
            if self._stop:
                break
            should, reason, fp = self.evaluate(folder)
            if not should:
                log.debug("[watch] 건너뜀: %s — %s", folder.name, reason)
                results.append(JobResult(folder.name, "skipped", reason))
                continue
            results.append(self.process(folder, fp))
        return results

    def stop(self) -> None:
        self._stop = True

    def run_forever(self) -> None:
        """중단 신호가 올 때까지 주기적으로 훑는다."""
        def handle(signum, frame):
            log.info("[watch] 종료 신호를 받았습니다. 진행 중인 작업을 마치고 멈춥니다.")
            self.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handle)
            except ValueError:
                pass  # 메인 스레드가 아니면 등록할 수 없다

        log.info("[watch] 감시 시작: %s → %s (%.0f초마다 확인)",
                 self.watch_dir, self.output_dir, self.poll_seconds)
        while not self._stop:
            try:
                for r in self.scan_once():
                    if r.status in ("done", "failed"):
                        log.info("[watch] %s: %s", r.name, r.status)
            except FileNotFoundError as e:
                log.error("[watch] %s", e)
            except Exception:
                # 한 번의 오류로 감시가 죽으면 안 된다. 학기 내내 도는 프로세스다.
                log.exception("[watch] 예상하지 못한 오류. 계속 감시합니다.")
            # 종료 신호에 빠르게 반응하도록 잘게 나눠 잔다
            slept = 0.0
            while slept < self.poll_seconds and not self._stop:
                time.sleep(min(1.0, self.poll_seconds - slept))
                slept += 1.0
        log.info("[watch] 감시를 마칩니다.")
