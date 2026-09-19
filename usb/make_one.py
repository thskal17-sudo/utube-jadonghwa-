#!/usr/bin/env python3
"""영상 한 편을 대화식으로 만든다 (USB 배포용).

명령줄 옵션을 외우지 않아도 되도록, 물어보고 만든다. 폴더 감시를 켜 두지 않고
그때그때 한 편씩 만들 때 쓴다.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n취소했습니다.")
        sys.exit(1)
    return answer or default


def pick_folder(watch: Path) -> Path:
    folders = sorted(p for p in watch.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not folders:
        print(f"\n  {watch} 안에 폴더가 없습니다.")
        print("  활동 이름으로 폴더를 만들고 사진을 넣은 뒤 다시 실행하세요.")
        sys.exit(1)

    print("\n  만들 수 있는 활동:\n")
    from shorts_pipeline.config import IMAGE_EXTENSIONS

    for i, folder in enumerate(folders, 1):
        n = sum(1 for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        print(f"    {i}. {folder.name}  (사진 {n}장)")

    while True:
        choice = ask("\n  번호를 고르세요", "1")
        if choice.isdigit() and 1 <= int(choice) <= len(folders):
            return folders[int(choice) - 1]
        print("  그 번호는 없습니다.")


def main() -> int:
    print()
    print("  " + "=" * 46)
    print("   교육사진 숏폼 만들기")
    print("  " + "=" * 46)

    watch = ROOT / "watch"
    watch.mkdir(exist_ok=True)
    folder = pick_folder(watch)

    from shorts_pipeline import ShortsConfig, run_pipeline
    from shorts_pipeline.watcher import read_description

    default_desc = read_description(folder)
    print()
    description = ask("  활동을 한 줄로 설명해 주세요", default_desc)

    print()
    print("  화질을 고르세요.")
    print("    1. 정식 (1080x1920, 2~3분 걸림)")
    print("    2. 미리보기 (540x960, 30초 정도)")
    preview = ask("  번호", "1") == "2"

    duration = ask("  영상 길이(초)", "18")
    try:
        duration = float(duration)
    except ValueError:
        duration = 18.0

    cfg = ShortsConfig(duration=duration)
    if preview:
        cfg = cfg.as_preview()

    output = ROOT / "output" / f"{folder.name}.mp4"
    print()
    print(f"  만드는 중입니다. 창을 닫지 마세요...")
    print()

    try:
        report = run_pipeline(folder, description, output, cfg,
                              work_dir=ROOT / "output" / f"{folder.name}_work")
    except FileNotFoundError as e:
        print(f"\n  오류: {e}")
        return 1
    except Exception:
        print("\n  예상하지 못한 오류가 났습니다. 아래 내용을 보내주시면 고쳐 드립니다.\n")
        traceback.print_exc()
        return 1

    print()
    print("  " + "=" * 46)
    print("   완성했습니다")
    print("  " + "=" * 46)
    print(f"   영상   : {report.output}")
    print(f"   제목   : {report.caption_plan['title']}")
    print(f"   사진   : {len(report.photos_used)}장, 가린 곳 {report.faces_per_photo}")
    print()
    if report.review_sheet:
        print("   업로드 전에 반드시 확인하세요:")
        print(f"   {report.review_sheet}")
        print("   초록 상자가 가려진 얼굴입니다. 빠진 얼굴이 없는지 눈으로 보세요.")
    for w in report.warnings:
        print(f"   주의: {w}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
