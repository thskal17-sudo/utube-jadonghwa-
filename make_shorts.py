#!/usr/bin/env python3
"""교육사진 → 유튜브 숏폼 mp4 (Phase 1 MVP) 명령줄 진입점.

예)
  python make_shorts.py ./photos/과학실험 -d "3학년 과학 시간, 화산 폭발 실험" -o out/volcano.mp4
  python make_shorts.py ./photos/운동회 -d "가을 운동회 이어달리기" --bgm energetic --title "운동회 하이라이트"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from shorts_pipeline import ShortsConfig, run_pipeline
from shorts_pipeline.bgm import STYLES
from shorts_pipeline.captions import CaptionPlan


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="사진 폴더 + 활동 설명 → 9:16 숏폼 mp4")
    p.add_argument("input_dir", type=Path, help="사진 폴더 경로")
    p.add_argument("-d", "--description", required=True, help="활동 설명 한 줄 (첫 장면 자막·제목에 사용)")
    p.add_argument("-o", "--output", type=Path, default=None, help="출력 mp4 경로 (기본: output/<폴더명>.mp4)")
    p.add_argument("--title", default=None, help="상단 제목 (기본: 설명을 다듬어 자동 생성)")
    p.add_argument("--captions", type=Path, default=None, help="장면별 자막 JSON (title/captions/hashtags)")
    p.add_argument("--duration", type=float, default=18.0, help="영상 길이(초), 15~20 권장")
    p.add_argument("--max-photos", type=int, default=8, help="사용할 최대 사진 수")
    p.add_argument("--bgm", choices=sorted(STYLES), default="bright", help="BGM 스타일")
    p.add_argument("--bgm-volume", type=float, default=0.35)
    p.add_argument("--font", type=Path, default=None, help="한글 폰트 파일(TTF/OTF)")
    p.add_argument("--detector", choices=["auto", "mediapipe", "yunet", "haar"], default="auto",
                   help="얼굴 감지기 (auto: mediapipe → yunet → haar 순으로 자동 선택)")
    p.add_argument("--yunet-model", type=Path, default=None)
    p.add_argument("--blur-method", choices=["pixelate", "gaussian"], default="pixelate")
    p.add_argument("--blur-padding", type=float, default=0.35)
    p.add_argument("--faces-json", type=Path, default=None,
                   help='놓친 얼굴 좌표 JSON {"사진.jpg": [[x,y,w,h], ...]}')
    p.add_argument("--no-blur", action="store_true", help="얼굴 블러 생략 (테스트 전용, 업로드 금지)")
    p.add_argument("--work-dir", type=Path, default=None, help="중간 산출물 폴더")
    p.add_argument("--seed", type=int, default=0, help="효과/BGM 변주 시드")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--preview", action="store_true", help="540x960 저해상도 빠른 확인용")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    output = args.output or Path("output") / f"{args.input_dir.resolve().name}.mp4"
    cfg = ShortsConfig(
        fps=args.fps,
        duration=args.duration,
        max_photos=args.max_photos,
        blur=not args.no_blur,
        detector=args.detector,
        yunet_model=args.yunet_model,
        blur_method=args.blur_method,
        blur_padding=args.blur_padding,
        manual_faces=args.faces_json,
        font_path=args.font,
        title=args.title,
        bgm_style=args.bgm,
        bgm_volume=args.bgm_volume,
        seed=args.seed,
    )
    if args.preview:
        cfg = cfg.as_preview()

    plan = CaptionPlan.from_json(args.captions) if args.captions else None

    try:
        report = run_pipeline(args.input_dir, args.description, output, cfg, work_dir=args.work_dir, caption_plan=plan)
    except (FileNotFoundError, ValueError) as e:
        print(f"오류: {e}", file=sys.stderr)
        return 1

    print("\n=== 완료 ===")
    print(f"출력      : {report.output} ({report.resolution}, {report.duration:.0f}s)")
    print(f"사진      : {len(report.photos_used)}장, 감지된 얼굴: {report.faces_per_photo} ({report.detector})")
    print(f"제목      : {report.caption_plan['title']}")
    print(f"소요 시간 : {report.seconds_elapsed}s")
    if report.review_sheet:
        print(f"검수 시트 : {report.review_sheet}  ← 업로드 전 얼굴 블러가 빠진 곳이 없는지 꼭 확인하세요")
    for w in report.warnings:
        print(f"주의      : {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
