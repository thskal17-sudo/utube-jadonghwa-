"""MediaPipe 버전 호환 계층.

mediapipe 0.10.30 부터 레거시 `mp.solutions` API 가 제거되고 Tasks API 만 남았습니다.
`mp.solutions.face_detection` / `mp.solutions.pose` 를 직접 부르면 최신 버전에서
`AttributeError: module 'mediapipe' has no attribute 'solutions'` 로 죽습니다.

이 모듈은 둘 중 어느 API 가 설치돼 있어도 같은 형식으로 쓸 수 있게 감쌉니다.

| API | 감지기 | 모델 |
| --- | --- | --- |
| legacy (`mp.solutions`, ~0.10.21) | `FaceDetection(model_selection=1)`, `Pose` | 패키지 내장 |
| tasks (`mp.tasks`, 0.10.30~) | `vision.FaceDetector`, `vision.PoseLandmarker` | 파일 필요 (`models/`) |

Tasks API 는 모델 파일을 패키지에 넣지 않으므로 `scripts/download_models.py` 로
두 개를 받아야 합니다. 얼굴 모델은 레거시 `model_selection=1`(전체 범위)이 쓰던
것과 같은 파일이라 감지 성능이 유지됩니다.

주의: Tasks API 는 리눅스에서 시스템 라이브러리 libGLESv2/libEGL 을 요구합니다.
(`sudo apt install libgles2 libegl1`)
"""

from __future__ import annotations

import atexit
import logging
import weakref
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# 열려 있는 Tasks 감지기들. 인터프리터 종료 시 네이티브 자원을 먼저 닫아 준다.
# 닫지 않으면 종료 단계에서 MediaPipe 의 __del__ 이 이미 해제된 C 바인딩을 만나
# "Exception ignored in ... TypeError: 'NoneType' object is not callable" 을 쏟는다.
_OPEN: "weakref.WeakSet" = weakref.WeakSet()


@atexit.register
def _close_all() -> None:
    for obj in list(_OPEN):
        try:
            obj.close()
        except Exception:  # 종료 중이므로 조용히 넘어간다
            pass


Box = Tuple[int, int, int, int]  # x, y, w, h

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# 레거시 model_selection=1 (전체 범위) 이 쓰는 것과 같은 모델
FACE_MODEL_NAME = "face_detection_full_range_sparse.tflite"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-assets/face_detection_full_range_sparse.tflite"
)
POSE_MODEL_NAME = "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# MediaPipe Pose 랜드마크 0~10 = 코, 양눈(6), 양귀, 입 양끝
HEAD_LANDMARK_COUNT = 11

GL_HINT = (
    "리눅스에서 MediaPipe Tasks API 는 시스템 라이브러리가 필요합니다: "
    "sudo apt install libgles2 libegl1"
)
DOWNLOAD_HINT = "python scripts/download_models.py 로 모델을 내려받으세요."


class MediaPipeUnavailable(RuntimeError):
    """MediaPipe 를 쓸 수 없는 상태(미설치·API 제거·모델 없음)."""


class _Closable:
    """네이티브 자원을 종료 시점에 확실히 닫기 위한 공통 기반."""

    _impl = None

    def _register(self) -> None:
        _OPEN.add(self)

    def close(self) -> None:
        impl, self._impl = self._impl, None
        _OPEN.discard(self)
        if impl is not None and hasattr(impl, "close"):
            try:
                impl.close()
            except Exception:  # 닫기 실패는 결과에 영향이 없다
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _mediapipe():
    try:
        import mediapipe as mp
    except ImportError:
        return None
    return mp


def mediapipe_installed() -> bool:
    return _mediapipe() is not None


def mediapipe_version() -> Optional[str]:
    mp = _mediapipe()
    return getattr(mp, "__version__", None) if mp else None


def has_legacy_solutions() -> bool:
    """레거시 `mp.solutions` API(모델 내장)를 쓸 수 있는지."""
    mp = _mediapipe()
    return mp is not None and hasattr(mp, "solutions")


def has_tasks_api() -> bool:
    """Tasks API 모듈을 임포트할 수 있는지(모델 파일 유무는 별도)."""
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
    except Exception:  # 임포트 단계에서 네이티브 로딩이 실패할 수도 있다
        return False
    return True


def face_model_path(explicit: Optional[Path] = None) -> Path:
    return Path(explicit) if explicit else MODELS_DIR / FACE_MODEL_NAME


def pose_model_path(explicit: Optional[Path] = None) -> Path:
    return Path(explicit) if explicit else MODELS_DIR / POSE_MODEL_NAME


def _model_ready(path: Path, min_bytes: int) -> bool:
    return path.exists() and path.stat().st_size >= min_bytes


def face_api_kind(model: Optional[Path] = None) -> Optional[str]:
    """얼굴 감지에 쓸 API 이름("legacy"/"tasks") 또는 불가하면 None."""
    if has_legacy_solutions():
        return "legacy"
    if has_tasks_api() and _model_ready(face_model_path(model), 100_000):
        return "tasks"
    return None


def pose_api_kind(model: Optional[Path] = None) -> Optional[str]:
    """자세 추정에 쓸 API 이름("legacy"/"tasks") 또는 불가하면 None."""
    if has_legacy_solutions():
        return "legacy"
    if has_tasks_api() and _model_ready(pose_model_path(model), 1_000_000):
        return "tasks"
    return None


def face_detection_ready(model: Optional[Path] = None) -> bool:
    return face_api_kind(model) is not None


def pose_ready(model: Optional[Path] = None) -> bool:
    return pose_api_kind(model) is not None


def unavailable_reason(model_path: Path) -> str:
    """왜 못 쓰는지 사람이 읽을 수 있는 설명."""
    if not mediapipe_installed():
        return "mediapipe 가 설치돼 있지 않습니다. `pip install mediapipe`"
    version = mediapipe_version() or "?"
    if not has_tasks_api():
        return (
            f"mediapipe {version} 에서 레거시 solutions API 와 Tasks API 를 모두 쓸 수 없습니다. "
            f"{GL_HINT}"
        )
    return (
        f"mediapipe {version} 에는 레거시 solutions API 가 없어 Tasks API 모델 파일이 필요합니다: "
        f"{model_path} 없음 — {DOWNLOAD_HINT}"
    )


def _clip_box(box: Tuple[float, float, float, float], w: int, h: int) -> Optional[Box]:
    """실수 좌표 박스를 이미지 안으로 자른다. 너무 작아지면 버린다."""
    x, y, bw, bh = box
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(w, int(round(x + bw))), min(h, int(round(y + bh)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


class FaceDetection(_Closable):
    """전체 범위(full-range) 얼굴 감지기. 레거시/Tasks 어느 쪽이든 같은 결과 형식."""

    def __init__(self, min_confidence: float = 0.5, model: Optional[Path] = None):
        self.min_confidence = float(min_confidence)
        self.kind = face_api_kind(model)
        if self.kind is None:
            raise MediaPipeUnavailable(unavailable_reason(face_model_path(model)))

        if self.kind == "legacy":
            mp = _mediapipe()
            # model_selection=1 → 전체 범위 모델. 단체 사진의 작은 얼굴에 강하다.
            self._impl = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=self.min_confidence
            )
            self._register()
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._mp = mp
            self._impl = vision.FaceDetector.create_from_options(
                vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(face_model_path(model))
                    ),
                    min_detection_confidence=self.min_confidence,
                )
            )
            self._register()
        except Exception as e:  # 네이티브 로딩 실패 등
            raise MediaPipeUnavailable(
                f"MediaPipe Tasks 얼굴 감지기 초기화 실패: {e}\n{GL_HINT}"
            ) from e

    def detect(self, img_bgr: np.ndarray) -> List[Box]:
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        raw: List[Tuple[float, float, float, float]] = []

        if self.kind == "legacy":
            result = self._impl.process(rgb)
            for det in result.detections or []:
                b = det.location_data.relative_bounding_box
                raw.append((b.xmin * w, b.ymin * h, b.width * w, b.height * h))
        else:
            image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
            )
            result = self._impl.detect(image)
            for det in result.detections or []:
                bb = det.bounding_box
                raw.append((bb.origin_x, bb.origin_y, bb.width, bb.height))

        boxes = [_clip_box(b, w, h) for b in raw]
        return [b for b in boxes if b is not None]



class PoseHead(_Closable):
    """자세 추정으로 머리 랜드마크(코·눈·귀·입)를 찾는다. 얼굴이 안 보여도 동작한다."""

    def __init__(self, min_confidence: float = 0.2, model: Optional[Path] = None):
        self.min_confidence = float(min_confidence)
        self.kind = pose_api_kind(model)
        if self.kind is None:
            raise MediaPipeUnavailable(unavailable_reason(pose_model_path(model)))

        if self.kind == "legacy":
            mp = _mediapipe()
            self._impl = mp.solutions.pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                min_detection_confidence=self.min_confidence,
            )
            self._register()
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            self._mp = mp
            self._impl = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(pose_model_path(model))
                    ),
                    num_poses=1,
                    min_pose_detection_confidence=self.min_confidence,
                )
            )
            self._register()
        except Exception as e:
            raise MediaPipeUnavailable(
                f"MediaPipe Tasks 자세 추정기 초기화 실패: {e}\n{GL_HINT}"
            ) from e

    def head_landmarks(self, img_bgr: np.ndarray) -> List[Tuple[float, float, float]]:
        """머리 랜드마크를 [(x픽셀, y픽셀, 신뢰도), ...] 로 돌려준다. 없으면 빈 목록."""
        h, w = img_bgr.shape[:2]
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out: List[Tuple[float, float, float]] = []

        if self.kind == "legacy":
            result = self._impl.process(rgb)
            landmarks = getattr(result.pose_landmarks, "landmark", None)
            if not landmarks:
                return []
            for i, lm in enumerate(landmarks):
                if i < HEAD_LANDMARK_COUNT:
                    out.append((lm.x * w, lm.y * h, float(getattr(lm, "visibility", 1.0))))
            return out

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
        )
        result = self._impl.detect(image)
        poses = result.pose_landmarks or []
        if not poses:
            return []
        for i, lm in enumerate(poses[0]):
            if i < HEAD_LANDMARK_COUNT:
                out.append((lm.x * w, lm.y * h, float(getattr(lm, "visibility", 1.0))))
        return out

