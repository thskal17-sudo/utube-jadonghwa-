#!/usr/bin/env python3
"""--faces-json 에 넣을 좌표를 눈으로 읽기 위한 격자 이미지를 만든다.

검수 시트에서 놓친 얼굴을 발견했을 때, 그 좌표를 알아내려면 눈금이 필요하다.
이 스크립트는 사진 위에 좌표 격자를 그려 준다. 격자를 보고 가릴 영역의
왼쪽 위 모서리(x, y)와 크기(너비, 높이)를 읽어 JSON 에 적으면 된다.

    python scripts/make_grid.py ./photos/수업/4.jpg
    # → 4_grid.jpg 생성. 열어서 눈금을 읽는다.

    # faces.json
    { "4.jpg": [[1400, 660, 360, 390]] }

    python make_shorts.py ./photos/수업 -d "..." --faces-json faces.json

--blurred 를 붙이면 현재 블러가 적용된 상태 위에 격자를 그려서,
어디가 아직 안 가려졌는지 바로 확인할 수 있다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shorts_pipeline.config import ShortsConfig  # noqa: E402
from shorts_pipeline.face_blur import blur_faces, load_image_rgb  # noqa: E402

GRID_COLOR = (0, 255, 255)


def draw_grid(img, step: int, scale: float):
    H, W = img.shape[:2]
    out = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for gx in range(0, W + 1, step):
        px = int(gx * scale)
        cv2.line(out, (px, 0), (px, out.shape[0]), GRID_COLOR, 1)
        cv2.putText(out, str(gx), (px + 3, 18), font, 0.5, GRID_COLOR, 1, cv2.LINE_AA)
    for gy in range(0, H + 1, step):
        py = int(gy * scale)
        cv2.line(out, (0, py), (out.shape[1], py), GRID_COLOR, 1)
        cv2.putText(out, str(gy), (4, py - 5), font, 0.5, GRID_COLOR, 1, cv2.LINE_AA)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="좌표 읽기용 격자 이미지 생성")
    p.add_argument("image", type=Path, help="사진 파일")
    p.add_argument("-o", "--output", type=Path, default=None, help="출력 경로 (기본: <이름>_grid.jpg)")
    p.add_argument("--step", type=int, default=100, help="격자 간격(픽셀)")
    p.add_argument("--width", type=int, default=1400, help="출력 이미지 가로 크기")
    p.add_argument("--blurred", action="store_true", help="현재 블러를 적용한 상태 위에 격자를 그린다")
    args = p.parse_args(argv)

    if not args.image.exists():
        print(f"오류: 파일이 없습니다: {args.image}", file=sys.stderr)
        return 1

    rgb = load_image_rgb(args.image)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if args.blurred:
        from shorts_pipeline.pipeline import build_detector

        detector = build_detector(ShortsConfig())
        boxes = detector.detect(img)
        img = blur_faces(img, boxes)
        print(f"현재 감지: {len(boxes)}개 영역")

    H, W = img.shape[:2]
    out = draw_grid(img, args.step, args.width / W)
    dst = args.output or args.image.with_name(f"{args.image.stem}_grid.jpg")
    cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"생성: {dst}  (원본 {W}x{H}, 격자 {args.step}px)")
    print('가릴 영역을 읽어 JSON 에 적으세요: {"%s": [[x, y, 너비, 높이]]}' % args.image.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
