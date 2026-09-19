"""검증 모드: 이미 가려진 사진에서 '아직 선명한 얼굴'만 찾아낸다.

선생님이 직접 블러 처리한 사진을 올릴 때 쓴다. 파이프라인은 블러를 새로
적용하지 않고, 감지기가 찾은 머리 영역마다 '이미 가려졌는지'를 판정해서
아직 선명한 곳만 빨간색으로 표시한다. 사람이 한 장 빠뜨리는 일을 잡기 위한
두 번째 눈이다.

판정 방법
얼굴 크기가 제각각이라 원본 해상도에서 재면 기준이 흔들린다. 그래서 영역을
96x96 으로 정규화한 뒤 두 가지를 잰다.

- 라플라시안 분산: 가우시안 블러와 단색 덮기에서 0 에 가깝게 떨어진다.
- 평탄 비율: 작은 창 안의 분산이 거의 0 인 픽셀 비율. 모자이크는 칸 내부가
  평평해서 높게 나온다. 모자이크는 칸 경계가 날카로워 라플라시안으로는
  안 잡히므로 이 지표가 따로 필요하다.

둘 중 하나라도 통과하면 '가려짐'으로 본다.

실측값 (실제 교실 사진의 머리 영역 기준, 정규화 후 중앙값)

| 상태            | 라플라시안 | 평탄비율 |
| --------------- | ---------- | -------- |
| 선명한 원본     | 864        | 0.24     |
| 가우시안 sigma8 | 2.4        | 0.60     |
| 모자이크 5칸    | 298        | 0.82     |
| 모자이크 9칸    | 338        | 0.70     |
| 단색 덮기       | 0          | 1.00     |

임계값은 일부러 엄격하게 잡았다. '가려졌다'고 잘못 판정하면 노출로 이어지지만,
'선명하다'고 잘못 판정하면 사람이 한 번 더 보는 것으로 끝나기 때문이다.
그래서 모자이크 16칸처럼 이목구비가 남는 약한 처리는 노출로 본다.

한계: 스티커나 이모지로 얼굴을 덮은 경우는 가장자리가 날카로워서 '선명함'으로
잘못 표시될 수 있다. 검수 시트에서 눈으로 확인하면 된다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import cv2
import numpy as np

from .face_blur import Box

log = logging.getLogger(__name__)

NORM_SIZE = 96
LAPLACIAN_MAX = 15.0   # 이보다 낮으면 가우시안/단색으로 가려진 것
FLAT_MIN = 0.62        # 이보다 높으면 모자이크로 가려진 것
FLAT_VAR_EPS = 4.0     # '평평하다'고 볼 국소 분산
MIN_SIDE = 12          # 이보다 작은 영역은 판정하지 않는다


@dataclass
class RegionCheck:
    """머리 영역 하나에 대한 판정 결과."""

    box: Box
    anonymized: bool
    laplacian: float
    flat_ratio: float
    too_small: bool = False

    @property
    def exposed(self) -> bool:
        return not self.anonymized


def region_metrics(roi: np.ndarray) -> tuple:
    """정규화한 영역의 (라플라시안 분산, 평탄 비율)."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.resize(gray, (NORM_SIZE, NORM_SIZE), interpolation=cv2.INTER_AREA)
    laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    k = 3
    mean = cv2.blur(gray, (k, k))
    sq = cv2.blur(gray * gray, (k, k))
    var = np.clip(sq - mean * mean, 0, None)
    flat_ratio = float((var < FLAT_VAR_EPS).mean())
    return laplacian, flat_ratio


def check_region(img_bgr: np.ndarray, box: Box) -> RegionCheck:
    """영역 하나가 이미 가려져 있는지 판정한다."""
    H, W = img_bgr.shape[:2]
    x, y, w, h = box
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    roi = img_bgr[y0:y1, x0:x1]
    if roi.size == 0 or min(x1 - x0, y1 - y0) < MIN_SIDE:
        # 너무 작아 판정이 불안정하다. 노출로 보고 사람이 확인하게 둔다.
        return RegionCheck(box, anonymized=False, laplacian=0.0, flat_ratio=0.0, too_small=True)
    laplacian, flat_ratio = region_metrics(roi)
    anonymized = laplacian < LAPLACIAN_MAX or flat_ratio > FLAT_MIN
    return RegionCheck(box, anonymized, laplacian, flat_ratio)


def verify_image(img_bgr: np.ndarray, boxes: Sequence[Box]) -> List[RegionCheck]:
    return [check_region(img_bgr, b) for b in boxes]


def split_checks(checks: Sequence[RegionCheck]):
    """(가려진 영역, 노출된 영역) 으로 나눈다."""
    protected = [c.box for c in checks if c.anonymized]
    exposed = [c.box for c in checks if c.exposed]
    return protected, exposed


@dataclass
class VerifyResult:
    """사진 한 장의 검증 결과."""

    protected: List[Box]   # 이미 가려진 영역
    exposed: List[Box]     # 선명한 얼굴 — 얼굴 감지기가 직접 잡았다. 확실하다.
    suspect: List[Box]     # 선명한 머리 추정 영역 — 손이나 사물일 수도 있다.

    @property
    def needs_fix(self) -> bool:
        return bool(self.exposed)


def verify_detection(img_bgr: np.ndarray, detection) -> VerifyResult:
    """감지 결과를 신뢰도별로 나눠 검증한다.

    얼굴 감지기가 잡은 영역이 선명하면 '확실한 노출'이다. 가려진 얼굴은
    얼굴 감지기에 잡히지 않기 때문이다(실측: 가려진 사진 3장에서 0·0·1개,
    같은 사진의 원본에서 4·3·4개).

    자세 추정만으로 잡힌 영역이 선명하면 '의심'으로 따로 둔다. 이쪽은 손이나
    인형을 사람으로 오인한 경우가 섞여 있어서, 노출로 단정하면 오탐이 쏟아진다.
    """
    face_checks = verify_image(img_bgr, detection.face_boxes)
    exposed = [c.box for c in face_checks if c.exposed]

    # 얼굴 감지기가 담당하지 않은 나머지 영역
    covered = list(detection.face_boxes)
    suspect: List[Box] = []
    protected: List[Box] = []
    for box in detection.boxes:
        if any(_overlaps(box, fb) for fb in covered):
            continue
        check = check_region(img_bgr, box)
        (suspect if check.exposed else protected).append(box)

    protected += [c.box for c in face_checks if c.anonymized]
    return VerifyResult(protected=protected, exposed=exposed, suspect=suspect)


def _overlaps(a: Box, b: Box, thresh: float = 0.4) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    smaller = max(1, min(aw * ah, bw * bh))
    return inter / smaller >= thresh
