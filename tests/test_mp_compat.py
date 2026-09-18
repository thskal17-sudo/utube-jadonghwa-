"""MediaPipe 버전 호환 계층 테스트.

어떤 mediapipe 가 설치돼 있어도(또는 없어도) 깨지지 않아야 하므로,
버전에 의존하는 항목은 건너뛰고 판별 로직을 검증한다.
"""

import numpy as np
import pytest

from shorts_pipeline import mp_compat
from shorts_pipeline.mp_compat import MediaPipeUnavailable

FACE_READY = mp_compat.face_detection_ready()
POSE_READY = mp_compat.pose_ready()


def test_head_landmark_count_matches_mediapipe_pose():
    # 코(1) + 눈 6 + 귀 2 + 입 2 = 11
    assert mp_compat.HEAD_LANDMARK_COUNT == 11


def test_model_paths_default_and_explicit(tmp_path):
    assert mp_compat.face_model_path().name == mp_compat.FACE_MODEL_NAME
    assert mp_compat.pose_model_path().name == mp_compat.POSE_MODEL_NAME
    custom = tmp_path / "custom.task"
    assert mp_compat.face_model_path(custom) == custom
    assert mp_compat.pose_model_path(custom) == custom


def test_api_kind_prefers_legacy(monkeypatch):
    monkeypatch.setattr(mp_compat, "has_legacy_solutions", lambda: True)
    assert mp_compat.face_api_kind() == "legacy"
    assert mp_compat.pose_api_kind() == "legacy"


def test_api_kind_none_when_tasks_model_missing(monkeypatch, tmp_path):
    """레거시 API 가 없고 모델 파일도 없으면 '사용 불가'여야 한다.

    임포트 성공만 보고 사용 가능이라고 판단하면 mediapipe 0.10.30+ 에서
    감지기 초기화 순간에 죽는다(이 저장소에서 실제로 났던 문제).
    """
    monkeypatch.setattr(mp_compat, "has_legacy_solutions", lambda: False)
    monkeypatch.setattr(mp_compat, "has_tasks_api", lambda: True)
    missing = tmp_path / "nope.task"
    assert mp_compat.face_api_kind(missing) is None
    assert mp_compat.pose_api_kind(missing) is None
    assert mp_compat.face_detection_ready(missing) is False
    assert mp_compat.pose_ready(missing) is False


def test_api_kind_tasks_when_model_present(monkeypatch, tmp_path):
    monkeypatch.setattr(mp_compat, "has_legacy_solutions", lambda: False)
    monkeypatch.setattr(mp_compat, "has_tasks_api", lambda: True)
    face = tmp_path / "face.tflite"
    face.write_bytes(b"0" * 200_000)
    pose = tmp_path / "pose.task"
    pose.write_bytes(b"0" * 2_000_000)
    assert mp_compat.face_api_kind(face) == "tasks"
    assert mp_compat.pose_api_kind(pose) == "tasks"


def test_truncated_model_file_is_not_ready(monkeypatch, tmp_path):
    """다운로드가 중간에 막혀 작은 파일이 남았을 때를 걸러낸다."""
    monkeypatch.setattr(mp_compat, "has_legacy_solutions", lambda: False)
    monkeypatch.setattr(mp_compat, "has_tasks_api", lambda: True)
    stub = tmp_path / "face.tflite"
    stub.write_bytes(b"<!DOCTYPE html>403 Forbidden")
    assert mp_compat.face_api_kind(stub) is None


def test_unavailable_reason_is_actionable(monkeypatch, tmp_path):
    monkeypatch.setattr(mp_compat, "mediapipe_installed", lambda: True)
    monkeypatch.setattr(mp_compat, "has_legacy_solutions", lambda: False)
    monkeypatch.setattr(mp_compat, "has_tasks_api", lambda: True)
    reason = mp_compat.unavailable_reason(tmp_path / "nope.task")
    assert "download_models.py" in reason

    monkeypatch.setattr(mp_compat, "mediapipe_installed", lambda: False)
    assert "pip install mediapipe" in mp_compat.unavailable_reason(tmp_path / "nope.task")


def test_face_detection_raises_when_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(mp_compat, "has_legacy_solutions", lambda: False)
    monkeypatch.setattr(mp_compat, "has_tasks_api", lambda: False)
    with pytest.raises(MediaPipeUnavailable):
        mp_compat.FaceDetection(model=tmp_path / "nope.tflite")
    with pytest.raises(MediaPipeUnavailable):
        mp_compat.PoseHead(model=tmp_path / "nope.task")


@pytest.mark.parametrize(
    "box,expected",
    [
        ((10.4, 20.6, 30.0, 40.0), (10, 21, 30, 40)),  # 반올림
        ((-20.0, -20.0, 60.0, 60.0), (0, 0, 40, 40)),  # 왼쪽/위로 넘침 → 자름
        ((90.0, 90.0, 40.0, 40.0), (90, 90, 10, 10)),  # 오른쪽/아래로 넘침 → 자름
        ((99.0, 99.0, 0.5, 0.5), None),  # 너무 작음 → 버림
    ],
)
def test_clip_box(box, expected):
    assert mp_compat._clip_box(box, 100, 100) == expected


@pytest.mark.skipif(not FACE_READY, reason="MediaPipe 얼굴 감지 준비 안 됨")
def test_face_detection_returns_boxes_in_bounds():
    img = np.full((240, 320, 3), 180, dtype=np.uint8)
    with mp_compat.FaceDetection(min_confidence=0.3) as fd:
        assert fd.kind in {"legacy", "tasks"}
        for x, y, w, h in fd.detect(img):
            assert 0 <= x and 0 <= y
            assert x + w <= 320 and y + h <= 240


@pytest.mark.skipif(not POSE_READY, reason="MediaPipe 자세 추정 준비 안 됨")
def test_pose_head_returns_landmark_triples():
    img = np.full((240, 320, 3), 180, dtype=np.uint8)
    with mp_compat.PoseHead() as ph:
        landmarks = ph.head_landmarks(img)
    assert isinstance(landmarks, list)
    assert len(landmarks) <= mp_compat.HEAD_LANDMARK_COUNT
    for item in landmarks:
        assert len(item) == 3


@pytest.mark.skipif(not FACE_READY, reason="MediaPipe 얼굴 감지 준비 안 됨")
def test_close_is_idempotent():
    fd = mp_compat.FaceDetection(min_confidence=0.3)
    fd.close()
    fd.close()  # 두 번 닫아도 예외가 없어야 한다
