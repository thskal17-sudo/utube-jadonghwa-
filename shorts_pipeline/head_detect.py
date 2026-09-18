"""사람 검출 + 자세 추정 기반 머리 영역 찾기.

교실 사진에서는 얼굴 감지기만으로는 부족하다. 학생들이 고개를 숙이거나
옆을 보거나 멀리 앉아 있으면 BlazeFace 같은 얼굴 감지기가 대부분 놓친다
(실측: 실제 교실 사진 5장에서 사람 45명 중 얼굴 감지 13개).

그래서 세 가지 신호를 합쳐 가릴 영역을 정한다.

1. 얼굴 감지 (MediaPipe, 전체 + 타일 분할 확대) — 잡히면 가장 정확하다.
2. 사람 검출(YOLO) 후 사람별 자세 추정 — 코·눈·귀 랜드마크로 머리를 찾는다.
   얼굴이 안 보여도 동작하므로 뒤돌아 앉은 학생도 잡는다.
3. 자세 추정마저 실패하면 사람 박스 상단에 머리 크기 사각형을 둔다.

셋을 합집합으로 합친다. 과잉 차단(얼굴 아닌 곳이 가려짐)은 보기에 아쉬울
뿐이지만, 누락은 개인정보 노출이라 비용이 전혀 다르다.

YOLO 가중치(models/yolov8n.pt)가 없으면 이 감지기는 쓸 수 없고,
파이프라인은 얼굴 전용 감지기로 자동 대체된다.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .face_blur import Box, merge_boxes

log = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_YOLO_NAME = "yolov8n.pt"

# MediaPipe Pose 랜드마크 0~10 = 코, 양눈(6), 양귀, 입 양끝
HEAD_LANDMARKS = tuple(range(11))


def yolo_weights_path(explicit: Optional[Path] = None) -> Path:
    return Path(explicit) if explicit else MODELS_DIR / DEFAULT_YOLO_NAME


def person_detector_available(weights: Optional[Path] = None) -> bool:
    """YOLO 가중치와 ultralytics 가 모두 준비돼 있는지."""
    if not yolo_weights_path(weights).exists():
        return False
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


class PersonHeadDetector:
    """얼굴 + 사람/자세를 합쳐 '가려야 할 머리 영역'을 돌려준다."""

    def __init__(
        self,
        weights: Optional[Path] = None,
        person_conf: float = 0.25,
        face_conf: float = 0.3,
        pose_visibility: float = 0.25,
        tiles: int = 3,
        tile_overlap: float = 0.25,
        tile_upscale: float = 2.0,
        head_expand: float = 1.15,
        fallback_scale: float = 0.62,
        yolo_imgsz: int = 1280,
    ):
        self.person_conf = person_conf
        self.face_conf = face_conf
        self.pose_visibility = pose_visibility
        self.tiles = tiles
        self.tile_overlap = tile_overlap
        self.tile_upscale = tile_upscale
        self.head_expand = head_expand
        self.fallback_scale = fallback_scale
        self.yolo_imgsz = yolo_imgsz
        self.backend = "person"

        path = yolo_weights_path(weights)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO 가중치를 찾지 못했습니다: {path}\n"
                "python scripts/download_models.py 로 내려받으세요."
            )
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("ultralytics 가 설치돼 있지 않습니다. `pip install ultralytics`") from e
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError("mediapipe 가 설치돼 있지 않습니다. `pip install mediapipe`") from e

        self._yolo = YOLO(str(path))
        self._mp = mp
        self.last_stats: dict = {}

    # ---------------- 얼굴 ----------------
    def _faces_whole(self, img: np.ndarray) -> List[Box]:
        H, W = img.shape[:2]
        fd = self._mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=self.face_conf
        )
        res = fd.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.detections:
            return []
        out = []
        for d in res.detections:
            b = d.location_data.relative_bounding_box
            out.append((int(b.xmin * W), int(b.ymin * H), int(b.width * W), int(b.height * H)))
        return out

    def _faces_tiled(self, img: np.ndarray) -> List[Box]:
        """타일로 잘라 확대 후 감지. 멀리 있는 작은 얼굴을 잡는다."""
        H, W = img.shape[:2]
        n = self.tiles
        th = int(H / n * (1 + self.tile_overlap))
        tw = int(W / n * (1 + self.tile_overlap))
        up = self.tile_upscale
        boxes: List[Box] = []
        for y0 in np.linspace(0, max(0, H - th), n).astype(int):
            for x0 in np.linspace(0, max(0, W - tw), n).astype(int):
                tile = img[y0 : y0 + th, x0 : x0 + tw]
                if tile.size == 0:
                    continue
                big = cv2.resize(tile, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
                fd = self._mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=self.face_conf
                )
                res = fd.process(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
                if not res.detections:
                    continue
                bh, bw = big.shape[:2]
                for d in res.detections:
                    b = d.location_data.relative_bounding_box
                    boxes.append((
                        int(b.xmin * bw / up) + int(x0),
                        int(b.ymin * bh / up) + int(y0),
                        int(b.width * bw / up),
                        int(b.height * bh / up),
                    ))
        return boxes

    # ---------------- 자세 → 머리 ----------------
    def _head_from_pose(self, crop: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        with self._mp.solutions.pose.Pose(
            static_image_mode=True, model_complexity=1, min_detection_confidence=0.2
        ) as pose:
            res = pose.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            return None
        ch, cw = crop.shape[:2]
        pts = [
            (lm.x * cw, lm.y * ch)
            for i, lm in enumerate(res.pose_landmarks.landmark)
            if i in HEAD_LANDMARKS and lm.visibility > self.pose_visibility
        ]
        if len(pts) < 2:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        # 랜드마크는 이목구비만 덮으므로 정수리와 턱까지 넉넉히 확장한다
        size = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = size * self.head_expand
        return (cx - half, cy - half * 1.05, half * 2, half * 2.1)

    def _head_from_pose_retry(self, crop: np.ndarray):
        """작고 흐린 사람은 확대해서 한 번 더 시도한다."""
        hb = self._head_from_pose(crop)
        if hb is not None:
            return hb, False
        h, w = crop.shape[:2]
        longest = max(h, w, 1)
        if longest < 900:
            up = min(3.0, 900 / longest)
            big = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            hb = self._head_from_pose(big)
            if hb is not None:
                return tuple(v / up for v in hb), True
        return None, False

    # ---------------- 통합 ----------------
    def detect(self, img_bgr: np.ndarray) -> List[Box]:
        H, W = img_bgr.shape[:2]
        faces = self._faces_whole(img_bgr) + self._faces_tiled(img_bgr)

        result = self._yolo.predict(
            img_bgr, classes=[0], conf=self.person_conf, imgsz=self.yolo_imgsz, verbose=False
        )[0]
        heads: List[Box] = []
        fallbacks: List[Box] = []
        retried = 0
        for b in result.boxes.xyxy.cpu().numpy():
            x0, y0, x1, y1 = (int(v) for v in b)
            bw, bh = x1 - x0, y1 - y0
            pad = int(0.08 * max(bw, bh))
            cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
            cx1, cy1 = min(W, x1 + pad), min(H, y1 + pad)
            crop = img_bgr[cy0:cy1, cx0:cx1]
            if crop.size == 0:
                continue
            hb, was_retry = self._head_from_pose_retry(crop)
            if hb is not None:
                retried += int(was_retry)
                hx, hy, hw, hh = hb
                heads.append((int(cx0 + hx), int(cy0 + hy), int(hw), int(hh)))
            else:
                s = min(bw * self.fallback_scale, bh * 0.45)
                fallbacks.append((int(x0 + bw / 2 - s / 2), int(y0 - s * 0.05), int(s), int(s * 1.1)))

        self.last_stats = dict(
            persons=len(result.boxes), faces=len(faces), pose_heads=len(heads),
            pose_retried=retried, fallbacks=len(fallbacks),
        )

        merged = merge_boxes([tuple(int(v) for v in b) for b in faces + heads + fallbacks], iou_thresh=0.35)
        clipped: List[Box] = []
        for x, y, w, h in merged:
            a, b_ = max(0, x), max(0, y)
            c, d = min(W, x + w), min(H, y + h)
            if c - a > 2 and d - b_ > 2:
                clipped.append((a, b_, c - a, d - b_))
        return clipped
