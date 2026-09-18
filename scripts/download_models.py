#!/usr/bin/env python3
"""감지 모델 내려받기.

1. YOLOv8n (사람 검출) — 교실 사진에서 누락을 줄이는 핵심 모델. 강력 권장.
   라이선스: AGPL-3.0 (Ultralytics). 학교 내부용으로 쓰는 데는 문제가 없지만,
   이 파이프라인을 외부에 배포할 계획이라면 라이선스를 확인하세요.
2. MediaPipe Tasks 모델 (얼굴 감지 + 자세 추정) — mediapipe 0.10.30 부터
   레거시 solutions API 가 사라지고 모델이 패키지에 포함되지 않습니다.
   그 버전에서는 이 두 파일이 있어야 얼굴·자세 신호를 쓸 수 있습니다.
   라이선스: Apache-2.0 (Google MediaPipe).
3. YuNet (얼굴 감지, 선택) — OpenCV Zoo, Apache-2.0.

레거시 API 를 쓰는 예전 mediapipe(~0.10.21)가 설치돼 있으면 2번은 건너뜁니다.
"""

from __future__ import annotations

import contextlib
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shorts_pipeline import mp_compat  # noqa: E402

MODELS = Path(__file__).resolve().parent.parent / "models"

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)


@contextlib.contextmanager
def _chdir(path: Path):
    """작업 디렉터리를 임시로 바꾼다.

    ultralytics 는 가중치를 현재 디렉터리에 받는다. 저장소 루트에서 실행하면
    루트에 yolov8n.pt 가 남으므로, models/ 안에서 받게 한다.
    """
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


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
    with _chdir(MODELS):
        model = YOLO("yolov8n.pt")  # 없으면 자동으로 받는다
    src = Path(model.ckpt_path) if getattr(model, "ckpt_path", None) else dest
    if src.exists() and src.resolve() != dest.resolve():
        dest.write_bytes(src.read_bytes())
    print(f"완료: {dest}")
    return dest.exists()


def _download(url: str, dest: Path, min_bytes: int) -> bool:
    """url 을 dest 로 내려받고 크기를 검증한다."""
    try:
        print(f"내려받는 중: {dest.name}")
        urllib.request.urlretrieve(url, dest)
    except OSError as e:
        print(f"  실패: {e}", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return False
    if dest.stat().st_size < min_bytes:
        print(f"  파일이 너무 작습니다({dest.stat().st_size}B). 다운로드가 막혔을 수 있습니다.", file=sys.stderr)
        dest.unlink(missing_ok=True)
        return False
    print(f"  완료: {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return True


def fetch_mediapipe_tasks() -> bool:
    """MediaPipe Tasks API 용 얼굴·자세 모델. 레거시 API 가 있으면 필요 없다."""
    if not mp_compat.mediapipe_installed():
        print("mediapipe 가 없어 건너뜁니다. (`pip install mediapipe`)")
        return False
    if mp_compat.has_legacy_solutions():
        print(f"mediapipe {mp_compat.mediapipe_version()} 는 모델 내장 버전입니다. Tasks 모델은 필요 없습니다.")
        return True

    MODELS.mkdir(parents=True, exist_ok=True)
    ok = True
    for url, name, min_bytes in (
        (mp_compat.FACE_MODEL_URL, mp_compat.FACE_MODEL_NAME, 100_000),
        (mp_compat.POSE_MODEL_URL, mp_compat.POSE_MODEL_NAME, 1_000_000),
    ):
        dest = MODELS / name
        if dest.exists() and dest.stat().st_size >= min_bytes:
            print(f"이미 있음: {dest}")
            continue
        ok = _download(url, dest, min_bytes) and ok
    return ok


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
    yolo_ok = fetch_yolo()
    mp_ok = fetch_mediapipe_tasks()
    fetch_yunet()  # 선택 사항이라 실패해도 전체 실패로 보지 않는다

    print()
    if not yolo_ok:
        print("사람 검출 모델을 준비하지 못했습니다. 얼굴 전용 감지기로 동작합니다.", file=sys.stderr)
        return 1
    if not mp_ok:
        print("MediaPipe 얼굴·자세 모델이 없어 사람 박스 기하 추정만 씁니다(정확도 하락).", file=sys.stderr)
        print("검수 시트를 특히 꼼꼼히 확인하세요.", file=sys.stderr)
        return 1
    print("준비 완료. 이제 `--detector auto` 가 사람 검출 기반으로 동작합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
