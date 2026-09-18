"""사람 검출 기반 머리 감지기 테스트.

모델 가중치가 없는 환경에서도 테스트가 깨지지 않도록, 가중치가 필요한 항목은
건너뛰고 순수 로직만 검증한다.
"""

import numpy as np
import pytest

from shorts_pipeline.config import ShortsConfig
from shorts_pipeline.head_detect import (
    HEAD_LANDMARKS,
    PersonHeadDetector,
    person_detector_available,
    yolo_weights_path,
)
from shorts_pipeline.pipeline import build_detector

HAS_MODEL = person_detector_available()


def test_head_landmarks_cover_face_points():
    # MediaPipe Pose 0~10 = 코, 눈 6점, 귀 2점, 입 2점
    assert HEAD_LANDMARKS == tuple(range(11))


def test_weights_path_default_and_explicit(tmp_path):
    assert yolo_weights_path().name == "yolov8n.pt"
    custom = tmp_path / "custom.pt"
    assert yolo_weights_path(custom) == custom


def test_missing_weights_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        PersonHeadDetector(weights=tmp_path / "nope.pt")


def test_person_detector_unavailable_for_missing_weights(tmp_path):
    assert person_detector_available(tmp_path / "nope.pt") is False


def test_build_detector_falls_back_when_weights_missing(tmp_path):
    cfg = ShortsConfig(detector="auto", yolo_weights=tmp_path / "nope.pt")
    det = build_detector(cfg)
    assert det.backend in {"mediapipe", "yunet", "haar"}


def test_build_detector_explicit_person_raises_without_weights(tmp_path):
    cfg = ShortsConfig(detector="person", yolo_weights=tmp_path / "nope.pt")
    with pytest.raises(FileNotFoundError):
        build_detector(cfg)


@pytest.mark.parametrize("backend", ["mediapipe", "haar"])
def test_build_detector_honours_explicit_face_backend(backend):
    det = build_detector(ShortsConfig(detector=backend))
    assert det.backend == backend


@pytest.mark.skipif(not HAS_MODEL, reason="YOLO 가중치 없음")
def test_person_detector_returns_boxes_inside_image():
    det = PersonHeadDetector()
    assert det.backend == "person"
    img = np.full((480, 640, 3), 200, dtype=np.uint8)
    boxes = det.detect(img)
    for x, y, w, h in boxes:
        assert x >= 0 and y >= 0
        assert x + w <= 640 and y + h <= 480
        assert w > 0 and h > 0


@pytest.mark.skipif(not HAS_MODEL, reason="YOLO 가중치 없음")
def test_person_detector_records_stats():
    det = PersonHeadDetector()
    det.detect(np.full((320, 320, 3), 180, dtype=np.uint8))
    for key in ("persons", "faces", "pose_heads", "fallbacks"):
        assert key in det.last_stats


def test_expand_box_grows_around_centre():
    from shorts_pipeline.head_detect import expand_box

    x, y, w, h = expand_box((100, 100, 100, 100), 2.0)
    assert (w, h) == (200, 200)
    assert (x + w / 2, y + h / 2) == (150, 150)  # 중심 유지


def test_expand_box_identity():
    from shorts_pipeline.head_detect import expand_box

    assert expand_box((10, 20, 30, 40), 1.0) == (10, 20, 30, 40)


def test_prefer_precise_keeps_coarse_when_face_is_much_smaller():
    """얼굴 박스가 훨씬 작으면 거친 박스를 남겨야 한다.

    실측에서 이걸 버렸더니 고개를 기울인 학생의 턱이 노출됐다.
    """
    from shorts_pipeline.head_detect import prefer_precise

    face = (100, 100, 20, 20)       # 거친 박스 면적의 4%
    coarse = (90, 90, 100, 100)
    kept = prefer_precise([face], [coarse])
    assert coarse in kept


def test_prefer_precise_drops_redundant_coarse():
    from shorts_pipeline.head_detect import prefer_precise

    face = (100, 100, 90, 90)       # 거친 박스와 크기가 비슷 → 중복
    coarse = (98, 98, 100, 100)
    kept = prefer_precise([face], [coarse])
    assert coarse not in kept
    assert face in kept


def test_prefer_precise_without_faces_keeps_everything():
    from shorts_pipeline.head_detect import prefer_precise

    coarse = [(0, 0, 10, 10), (50, 50, 20, 20)]
    assert prefer_precise([], coarse) == coarse
