"""1단계: 얼굴 감지 + 블러(모자이크) 처리.

감지기는 세 가지를 지원하고, backend="auto" 면 정확도 순으로 자동 선택한다.

1. mediapipe : BlazeFace. 모델이 패키지에 포함돼 있어 추가 다운로드가 필요 없다.
               측면·작은 얼굴 감지율이 가장 높아 기본값으로 권장한다.
2. yunet     : ONNX 모델. models/ 에 파일이 있으면 사용 (scripts/download_models.py).
3. haar      : OpenCV 내장 Haar cascade. 항상 사용 가능한 최후의 대비책이지만
               측면·작은 얼굴을 자주 놓친다.

주의: 어떤 감지기도 100% 정확하지 않다. 파이프라인이 만드는 검수 시트
(review_sheet.jpg)를 업로드 전에 반드시 육안으로 확인할 것. 놓친 얼굴은
--faces-json 으로 직접 좌표를 지정해 가릴 수 있다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # x, y, w, h

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_YUNET_NAME = "face_detection_yunet_2023mar.onnx"


@dataclass
class FaceBlurResult:
    source: Path
    output: Path
    boxes: List[Box] = field(default_factory=list)
    detector: str = "haar"

    @property
    def face_count(self) -> int:
        return len(self.boxes)


def load_image_rgb(path: Path) -> np.ndarray:
    """EXIF 회전을 반영해 RGB numpy 배열로 읽는다."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB"))


def _iou(a: Box, b: Box) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def merge_boxes(boxes: Sequence[Box], iou_thresh: float = 0.3) -> List[Box]:
    """여러 감지기 결과 중 겹치는 박스를 합집합으로 합친다."""
    kept: List[Box] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        for i, k in enumerate(kept):
            if _iou(box, k) >= iou_thresh:
                x0 = min(box[0], k[0])
                y0 = min(box[1], k[1])
                x1 = max(box[0] + box[2], k[0] + k[2])
                y1 = max(box[1] + box[3], k[1] + k[3])
                kept[i] = (x0, y0, x1 - x0, y1 - y0)
                break
        else:
            kept.append(tuple(int(v) for v in box))  # type: ignore[arg-type]
    return kept


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return True


class FaceDetector:
    """MediaPipe / YuNet / Haar 중 하나를 사용하는 얼굴 감지기."""

    def __init__(
        self,
        backend: str = "auto",
        yunet_model: Optional[Path] = None,
        min_face_ratio: float = 0.03,
        score_threshold: float = 0.5,
        max_side: int = 1280,
    ):
        self.min_face_ratio = min_face_ratio
        self.score_threshold = score_threshold
        self.max_side = max_side
        self._yunet = None
        self._mp = None
        self.backend = self._select_backend(backend, yunet_model)

        if self.backend == "mediapipe":
            self._init_mediapipe()
        elif self.backend == "yunet":
            self._init_yunet(yunet_model)
        else:
            self._init_haar()
        log.info("[detector] %s 감지기를 사용합니다.", self.backend)

    # -- 백엔드 선택 --------------------------------------------------
    def _yunet_path(self, yunet_model: Optional[Path]) -> Path:
        return Path(yunet_model) if yunet_model else MODELS_DIR / DEFAULT_YUNET_NAME

    def _yunet_ready(self, yunet_model: Optional[Path]) -> bool:
        p = self._yunet_path(yunet_model)
        return p.exists() and p.stat().st_size > 10_000 and hasattr(cv2, "FaceDetectorYN")

    def _select_backend(self, backend: str, yunet_model: Optional[Path]) -> str:
        if backend != "auto":
            return backend
        if mediapipe_available():
            return "mediapipe"
        if self._yunet_ready(yunet_model):
            return "yunet"
        log.warning(
            "MediaPipe 와 YuNet 이 모두 없어 Haar cascade 로 동작합니다. "
            "감지율이 낮으니 `pip install mediapipe` 를 권장합니다."
        )
        return "haar"

    # -- 백엔드별 초기화 ----------------------------------------------
    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as e:
            raise RuntimeError("mediapipe 가 설치돼 있지 않습니다. `pip install mediapipe`") from e
        # model_selection=1 → full range 모델. 단체 사진의 작은 얼굴에 강하다.
        self._mp = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=self.score_threshold
        )

    def _init_yunet(self, yunet_model: Optional[Path]) -> None:
        model = self._yunet_path(yunet_model)
        if not self._yunet_ready(yunet_model):
            raise FileNotFoundError(
                f"YuNet 모델을 찾지 못했습니다: {model}\n"
                "python scripts/download_models.py 로 내려받으세요."
            )
        self._yunet = cv2.FaceDetectorYN.create(
            str(model), "", (320, 320), self.score_threshold, 0.3, 5000
        )

    def _init_haar(self) -> None:
        base = cv2.data.haarcascades
        self._cascades = [
            cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml"),
            cv2.CascadeClassifier(base + "haarcascade_frontalface_alt2.xml"),
        ]
        self._profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
        if any(c.empty() for c in self._cascades) or self._profile.empty():
            raise RuntimeError("OpenCV Haar cascade 파일을 불러오지 못했습니다.")

    # ------------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray) -> List[Box]:
        h, w = img_bgr.shape[:2]
        scale = 1.0
        work = img_bgr
        if max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
            work = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        if self.backend == "mediapipe":
            boxes = self._detect_mediapipe(work)
        elif self.backend == "yunet":
            boxes = self._detect_yunet(work)
        else:
            boxes = self._detect_haar(work)

        if scale != 1.0:
            boxes = [tuple(int(round(v / scale)) for v in b) for b in boxes]  # type: ignore[misc]
        return merge_boxes(boxes)

    def _detect_mediapipe(self, img: np.ndarray) -> List[Box]:
        h, w = img.shape[:2]
        result = self._mp.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not result.detections:
            return []
        boxes: List[Box] = []
        for det in result.detections:
            bb = det.location_data.relative_bounding_box
            x, y = int(bb.xmin * w), int(bb.ymin * h)
            bw, bh = int(bb.width * w), int(bb.height * h)
            # 상대 좌표는 화면 밖으로 조금 넘칠 수 있다.
            x, y = max(0, x), max(0, y)
            bw, bh = min(bw, w - x), min(bh, h - y)
            if bw > 1 and bh > 1:
                boxes.append((x, y, bw, bh))
        return boxes

    def _detect_yunet(self, img: np.ndarray) -> List[Box]:
        h, w = img.shape[:2]
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(img)
        if faces is None:
            return []
        return [tuple(int(v) for v in f[:4]) for f in faces]  # type: ignore[misc]

    def _detect_haar(self, img: np.ndarray) -> List[Box]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        h, w = gray.shape
        min_size = max(24, int(min(h, w) * self.min_face_ratio))
        boxes: List[Box] = []

        for cascade in self._cascades:
            found = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
            )
            boxes.extend(tuple(int(v) for v in f) for f in found)  # type: ignore[misc]

        # 측면 얼굴: 원본 + 좌우 반전
        found = self._profile.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6, minSize=(min_size, min_size)
        )
        boxes.extend(tuple(int(v) for v in f) for f in found)  # type: ignore[misc]
        flipped = cv2.flip(gray, 1)
        found = self._profile.detectMultiScale(
            flipped, scaleFactor=1.1, minNeighbors=6, minSize=(min_size, min_size)
        )
        for (x, y, bw, bh) in found:
            boxes.append((int(w - x - bw), int(y), int(bw), int(bh)))
        return boxes


# ----------------------------------------------------------------------
def blur_faces(
    img_bgr: np.ndarray,
    boxes: Sequence[Box],
    method: str = "pixelate",
    padding: float = 0.35,
    strength: float = 1.0,
) -> np.ndarray:
    """감지된 얼굴 영역을 타원형으로 모자이크/블러 처리한 사본을 돌려준다."""
    out = img_bgr.copy()
    H, W = out.shape[:2]
    for (x, y, w, h) in boxes:
        px, py = int(w * padding), int(h * padding)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(W, x + w + px), min(H, y + h + py)
        rw, rh = x1 - x0, y1 - y0
        if rw < 2 or rh < 2:
            continue
        roi = out[y0:y1, x0:x1]

        if method == "gaussian":
            k = int(max(21, min(rw, rh) * 0.8 * strength)) | 1
            blurred = cv2.GaussianBlur(roi, (k, k), 0)
        else:  # pixelate
            # 얼굴 영역을 가로·세로 약 5칸으로 나눈다. 칸이 작을수록(=많을수록)
            # 이목구비가 남아 재식별 위험이 커지므로 의도적으로 크게 잡는다.
            block = max(4, int(min(rw, rh) / (5.0 * strength)))
            small = cv2.resize(roi, (max(1, rw // block), max(1, rh // block)), interpolation=cv2.INTER_LINEAR)
            blurred = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)

        mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2, rh // 2), 0, 0, 360, 255, -1)
        feather = max(1.0, min(rw, rh) * 0.04)
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        roi[:] = (blurred.astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    return out


def draw_boxes(img_bgr: np.ndarray, boxes: Sequence[Box], color=(0, 255, 0), thickness: int = 3) -> np.ndarray:
    out = img_bgr.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def make_review_sheet(results: Sequence[FaceBlurResult], path: Path, thumb_w: int = 480, cols: int = 3) -> Path:
    """블러 결과를 격자로 모아 검수용 이미지를 만든다(초록 박스 = 블러 적용 위치)."""
    tiles = []
    for r in results:
        img = cv2.imread(str(r.output))
        if img is None:
            continue
        img = draw_boxes(img, r.boxes)
        h, w = img.shape[:2]
        th = int(h * thumb_w / w)
        tile = cv2.resize(img, (thumb_w, th), interpolation=cv2.INTER_AREA)
        label = f"{r.source.name[:28]}  faces={r.face_count}"
        cv2.rectangle(tile, (0, 0), (thumb_w, 30), (0, 0, 0), -1)
        cv2.putText(tile, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    if not tiles:
        return path

    max_h = max(t.shape[0] for t in tiles)
    padded = [cv2.copyMakeBorder(t, 0, max_h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)) for t in tiles]
    rows = []
    for i in range(0, len(padded), cols):
        row = padded[i : i + cols]
        while len(row) < cols:
            row.append(np.full((max_h, thumb_w, 3), 30, dtype=np.uint8))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def load_manual_boxes(path: Path) -> dict:
    """수동 블러 좌표 JSON 을 읽는다.

    형식: {"사진파일명.jpg": [[x, y, w, h], ...], ...}
    감지기가 놓친 얼굴을 검수 후 직접 가릴 때 사용한다.
    """
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    out = {}
    for name, boxes in raw.items():
        parsed = []
        for b in boxes:
            if len(b) != 4:
                raise ValueError(f"{name}: 박스는 [x, y, w, h] 4개 값이어야 합니다 → {b}")
            parsed.append(tuple(int(v) for v in b))
        out[Path(name).name] = parsed
    return out


def blur_folder(
    image_paths: Sequence[Path],
    out_dir: Path,
    detector: Optional[FaceDetector] = None,
    method: str = "pixelate",
    padding: float = 0.35,
    strength: float = 1.0,
    review_path: Optional[Path] = None,
    manual_boxes: Optional[dict] = None,
) -> List[FaceBlurResult]:
    """사진 목록을 모두 블러 처리해 out_dir 에 저장한다."""
    detector = detector or FaceDetector()
    manual_boxes = manual_boxes or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[FaceBlurResult] = []
    for i, src in enumerate(image_paths):
        rgb = load_image_rgb(src)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        boxes = detector.detect(bgr)
        extra = manual_boxes.get(src.name, [])
        if extra:
            boxes = merge_boxes(list(boxes) + list(extra))
            log.info("[blur] %s ← 수동 지정 %d개 추가", src.name, len(extra))
        blurred = blur_faces(bgr, boxes, method=method, padding=padding, strength=strength)
        dst = out_dir / f"{i:03d}_{src.stem}_blur.jpg"
        cv2.imwrite(str(dst), blurred, [cv2.IMWRITE_JPEG_QUALITY, 95])
        results.append(FaceBlurResult(source=src, output=dst, boxes=boxes, detector=detector.backend))
        log.info("[blur] %s → 얼굴 %d개", src.name, len(boxes))
    if review_path:
        make_review_sheet(results, review_path)
    return results
