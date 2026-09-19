#!/usr/bin/env python3
"""폴더를 감시해 사진이 들어오면 자동으로 숏폼을 만든다 (Phase 4).

    watch/
    ├── 과학실험/            ← 영상 한 편이 된다
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── description.txt  (선택: 활동 설명. 없으면 폴더명을 쓴다)
    └── 운동회/

사용 예)
    # 계속 감시 (학기 중 켜 두기)
    python watch_shorts.py ./watch -o ./output

    # 한 번만 훑기 (cron 이나 작업 스케줄러에 등록)
    python watch_shorts.py ./watch -o ./output --once

결과는 output/ 에 저장되고 거기서 끝납니다. 유튜브에 자동으로 올라가지 않습니다.
얼굴 블러가 완벽하지 않고 자막이 생성형이라, 사람이 보지 않은 영상이 채널에
올라가면 안 되기 때문입니다.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from shorts_pipeline import ShortsConfig
from shorts_pipeline.bgm import STYLES
from shorts_pipeline.watcher import (
    DEFAULT_POLL_SECONDS,
    DEFAULT_QUIET_SECONDS,
    LOCK_FILENAME,
    FolderWatcher,
    LockError,
    SingleInstanceLock,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="사진 폴더를 감시해 자동으로 숏폼을 만듭니다",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("watch_dir", type=Path, help="감시할 상위 폴더 (하위 폴더 하나 = 영상 한 편)")
    p.add_argument("-o", "--output", type=Path, default=Path("output"), help="결과를 저장할 폴더")
    p.add_argument("--once", action="store_true", help="한 번만 훑고 끝냅니다 (cron 용)")
    p.add_argument("--poll", type=float, default=DEFAULT_POLL_SECONDS, help="확인 주기(초)")
    p.add_argument("--quiet-seconds", type=float, default=DEFAULT_QUIET_SECONDS,
                   help="이 시간 동안 파일 변화가 없으면 복사가 끝났다고 봅니다")
    p.add_argument("--min-photos", type=int, default=2, help="영상을 만들 최소 사진 수")
    p.add_argument("--reprocess", action="store_true",
                   help="이미 처리했거나 실패한 폴더도 다시 만듭니다")
    p.add_argument("--dry-run", action="store_true",
                   help="무엇을 처리할지 보여주기만 하고 만들지는 않습니다")

    g = p.add_argument_group("영상 설정 (make_shorts.py 와 동일)")
    g.add_argument("--duration", type=float, default=18.0)
    g.add_argument("--max-photos", type=int, default=8)
    g.add_argument("--bgm", choices=sorted(STYLES), default="bright")
    g.add_argument("--font", type=Path, default=None)
    g.add_argument("--detector", choices=["auto", "person", "mediapipe", "yunet", "haar"], default="auto")
    g.add_argument("--ai-captions", action="store_true", help="Claude 가 사진을 보고 자막 생성")
    g.add_argument("--preview", action="store_true", help="540x960 저화질 (빠른 확인용)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = ShortsConfig(
        duration=args.duration,
        max_photos=args.max_photos,
        bgm_style=args.bgm,
        font_path=args.font,
        detector=args.detector,
        ai_captions=args.ai_captions,
    )
    if args.preview:
        cfg = cfg.as_preview()

    watcher = FolderWatcher(
        watch_dir=args.watch_dir,
        output_dir=args.output,
        config=cfg,
        quiet_seconds=args.quiet_seconds,
        poll_seconds=args.poll,
        min_photos=args.min_photos,
        reprocess=args.reprocess,
    )

    if args.dry_run:
        try:
            folders = watcher.job_folders()
        except FileNotFoundError as e:
            print(f"오류: {e}", file=sys.stderr)
            return 1
        if not folders:
            print(f"{args.watch_dir} 아래에 하위 폴더가 없습니다.")
            return 0
        print(f"{args.watch_dir} 아래 {len(folders)}개 폴더:\n")
        for folder in folders:
            should, reason, fp = watcher.evaluate(folder)
            mark = "처리함" if should else "건너뜀"
            detail = f"사진 {fp.count}장" if should else reason
            print(f"  [{mark}] {folder.name}  — {detail}")
        return 0

    try:
        with SingleInstanceLock(args.output / LOCK_FILENAME):
            if args.once:
                results = watcher.scan_once()
                done = [r for r in results if r.status == "done"]
                failed = [r for r in results if r.status == "failed"]
                print(f"\n처리 {len(done)}건, 실패 {len(failed)}건, "
                      f"건너뜀 {len(results) - len(done) - len(failed)}건")
                for r in done:
                    print(f"  완료: {r.name} → {r.output}  (검수 필요)")
                for r in failed:
                    print(f"  실패: {r.name} — {r.message}")
                return 1 if failed else 0
            watcher.run_forever()
            return 0
    except LockError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
