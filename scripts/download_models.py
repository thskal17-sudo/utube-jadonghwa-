#!/usr/bin/env python3
"""선택 사항: YuNet 얼굴 감지 모델(ONNX, Apache-2.0) 내려받기.

Haar cascade 보다 측면·작은 얼굴 감지율이 훨씬 높다. 받아 두면
FaceDetector(backend="auto") 가 자동으로 우선 사용한다.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)
DEST = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"다운로드: {URL}\n    → {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    size = DEST.stat().st_size
    if size < 100_000:
        print(f"경고: 파일 크기가 너무 작습니다({size} bytes). 다운로드가 막혔을 수 있습니다.", file=sys.stderr)
        DEST.unlink(missing_ok=True)
        return 1
    print(f"완료 ({size/1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
