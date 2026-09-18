#!/usr/bin/env python3
"""감지 모델 내려받기.

1. YOLOv8n (사람 검출) — 교실 사진에서 누락을 줄이는 핵심 모델. 강력 권장.
   라이선스: AGPL-3.0 (Ultralytics). 학교 내부용으로 쓰는 데는 문제가 없지만,
   이 파이프라인을 외부에 배포할 계획이라면 라이선스를 확인하세요.
2. YuNet (얼굴 감지, 선택) — OpenCV Zoo, Apache-2.0.

MediaPipe 모델은 패키지에 포함돼 있어 내려받을 필요가 없습니다.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


def fetch_yolo() -> bool:
    dest = MODELS / "yolov8n.pt"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"이미 있음: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
        return True
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics 가 없습니다. `pip install ultralytics` 후 다시 실행하세요.", file=sys.stderr)
        return False
    MODELS.mkdir(parents=True, exist_ok=True)
    print("YOLOv8n 내려받는 중 (사람 검출)...")
    model = YOLO("yolov8n.pt")  # 없으면 자동으로 받는다
    src = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else Path("yolov8n.pt")
    if src.exists() and src.resolve() != dest.resolve():
        dest.write_bytes(src.read_bytes())
    print(f"완료: {dest}")
    return dest.exists()


def fetch_yunet() -> bool:
    dest = MODELS / "face_detection_yunet_2023mar.onnx"
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"이미 있음: {dest}")
        return True
    MODELS.mkdir(parents=True, exist_ok=True)
    try:
        print(f"YuNet 내려받는 중: {YUNET_URL}")
        urllib.request.urlretrieve(YUNET_URL, dest)
    except OSError as e:
        print(f"YuNet 다운로드 실패(선택 사항이므로 건너뜁니다): {e}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return False
    if dest.stat().st_size < 100_000:
        print("YuNet 파일이 너무 작습니다. 다운로드가 막혔을 수 있습니다.", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return False
    print(f"완료: {dest}")
    return True


def main() -> int:
    ok = fetch_yolo()
    fetch_yunet()  # 선택 사항이라 실패해도 전체 실패로 보지 않는다
    if not ok:
        print("\n사람 검출 모델을 준비하지 못했습니다. 얼굴 전용 감지기로 동작합니다.", file=sys.stderr)
        return 1
    print("\n준비 완료. 이제 `--detector auto` 가 사람 검출 기반으로 동작합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
