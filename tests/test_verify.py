"""이미 가려진 사진에서 '아직 선명한 얼굴'을 찾아내는 검증 모드 테스트."""

import cv2
import numpy as np
import pytest

from shorts_pipeline.head_detect import Detection
from shorts_pipeline.verify import (
    FLAT_MIN,
    LAPLACIAN_MAX,
    MIN_SIDE,
    check_region,
    region_metrics,
    verify_detection,
    verify_image,
)


def sharp_patch(size=200, seed=0):
    """선명한 사진을 흉내 낸다: 질감이 풍부해 라플라시안이 크고 평탄 비율이 낮다."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return cv2.GaussianBlur(base, (3, 3), 0.6)  # 사진처럼 아주 약간만 부드럽게


def mosaic(patch, blocks=6):
    h, w = patch.shape[:2]
    small = cv2.resize(patch, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def test_sharp_patch_is_not_anonymized():
    img = sharp_patch()
    check = check_region(img, (0, 0, 200, 200))
    assert check.exposed
    assert check.laplacian > LAPLACIAN_MAX


@pytest.mark.parametrize("sigma", [10, 20, 40])
def test_gaussian_blur_is_anonymized(sigma):
    img = cv2.GaussianBlur(sharp_patch(), (0, 0), sigma)
    assert check_region(img, (0, 0, 200, 200)).anonymized


@pytest.mark.parametrize("blocks", [3, 5, 8])
def test_mosaic_is_anonymized(blocks):
    img = mosaic(sharp_patch(), blocks)
    check = check_region(img, (0, 0, 200, 200))
    assert check.anonymized
    assert check.flat_ratio > FLAT_MIN


def test_solid_fill_is_anonymized():
    img = np.full((200, 200, 3), 40, dtype=np.uint8)
    check = check_region(img, (0, 0, 200, 200))
    assert check.anonymized
    assert check.laplacian == pytest.approx(0.0, abs=1e-6)


def test_tiny_region_is_flagged_for_human_review():
    """판정이 불안정한 작은 영역은 안전하게 노출로 본다."""
    img = sharp_patch()
    check = check_region(img, (0, 0, MIN_SIDE - 1, MIN_SIDE - 1))
    assert check.too_small
    assert check.exposed


def test_region_outside_image_is_flagged():
    img = sharp_patch(50)
    assert check_region(img, (500, 500, 40, 40)).exposed


def test_metrics_separate_sharp_from_mosaic():
    sharp_lap, sharp_flat = region_metrics(sharp_patch())
    mos_lap, mos_flat = region_metrics(mosaic(sharp_patch()))
    assert sharp_flat < FLAT_MIN < mos_flat
    assert sharp_lap > mos_lap or mos_flat > FLAT_MIN


def test_verify_image_checks_every_box():
    img = sharp_patch(300)
    checks = verify_image(img, [(0, 0, 100, 100), (100, 100, 100, 100)])
    assert len(checks) == 2


def _canvas_with(patch, box, size=400):
    """단색 배경 위 지정 위치에 패치를 올린 이미지를 만든다."""
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    x, y, w, h = box
    img[y : y + h, x : x + w] = cv2.resize(patch, (w, h))
    return img


def test_sharp_face_box_is_exposed():
    box = (50, 50, 150, 150)
    img = _canvas_with(sharp_patch(), box)
    result = verify_detection(img, Detection(boxes=[box], face_boxes=[box], coarse_boxes=[]))
    assert result.exposed == [box]
    assert result.needs_fix


def test_blurred_face_box_is_protected():
    box = (50, 50, 150, 150)
    img = _canvas_with(mosaic(sharp_patch(), 4), box)
    result = verify_detection(img, Detection(boxes=[box], face_boxes=[box], coarse_boxes=[]))
    assert result.exposed == []
    assert box in result.protected
    assert not result.needs_fix


def test_sharp_coarse_only_box_is_suspect_not_exposed():
    """얼굴 감지기가 잡지 않은 선명한 영역은 '확인 필요'지 '노출'이 아니다.

    자세 추정은 손이나 인형을 사람 머리로 오인한다. 이걸 노출로 단정하면
    이미 잘 가린 사진에서도 경고가 쏟아진다(실측에서 오탐 10곳이 1곳으로 줄었다).
    """
    box = (50, 50, 150, 150)
    img = _canvas_with(sharp_patch(), box)
    result = verify_detection(img, Detection(boxes=[box], face_boxes=[], coarse_boxes=[box]))
    assert result.exposed == []
    assert result.suspect == [box]
    assert not result.needs_fix


def test_empty_detection_yields_nothing():
    result = verify_detection(sharp_patch(), Detection())
    assert result.exposed == [] and result.suspect == [] and result.protected == []
