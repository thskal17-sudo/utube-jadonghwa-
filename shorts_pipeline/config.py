"""파이프라인 전역 설정."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# 유튜브 숏폼 권장 규격
SHORTS_WIDTH = 1080
SHORTS_HEIGHT = 1920
SHORTS_MIN_SECONDS = 15.0
SHORTS_MAX_SECONDS = 20.0


@dataclass
class ShortsConfig:
    """한 번의 렌더링에 쓰이는 모든 설정값."""

    # 출력 규격
    width: int = SHORTS_WIDTH
    height: int = SHORTS_HEIGHT
    fps: int = 30
    duration: float = 18.0  # 최종 영상 길이(초). 15~20초 권장
    max_photos: int = 8  # 이보다 많으면 균등 간격으로 골라 씀
    transition: float = 0.6  # 장면 간 크로스페이드(초)

    # 켄 번즈 효과
    zoom_min: float = 1.0
    zoom_max: float = 1.18

    # 얼굴 블러
    blur: bool = True
    detector: str = "auto"  # auto | person | mediapipe | yunet | haar
    yunet_model: Optional[Path] = None
    yolo_weights: Optional[Path] = None
    blur_method: str = "pixelate"  # pixelate | gaussian
    blur_padding: float = 0.35  # 감지 박스를 이만큼(비율) 넓혀서 블러
    blur_strength: float = 1.0
    manual_faces: Optional[Path] = None  # 놓친 얼굴 좌표 JSON
    tight_blur: bool = False  # 차단 영역을 좁힘. 보기 좋지만 누락 위험이 커진다
    keep_faces: tuple = ()  # 블러를 적용하지 않을 사진 파일명들
    verify: bool = False  # 이미 가려진 사진을 검사만 한다(블러 재적용 안 함)

    # 자막
    font_path: Optional[Path] = None
    title: Optional[str] = None
    ai_captions: bool = False           # Claude 가 사진을 보고 자막을 생성
    ai_model: str = "claude-opus-5"
    ai_effort: str = "medium"

    # BGM
    bgm_style: str = "bright"  # bright | calm | energetic
    bgm_volume: float = 0.35
    seed: int = 0

    # 인코딩
    video_bitrate: str = "8000k"
    preset: str = "medium"
    threads: int = 4

    def as_preview(self) -> "ShortsConfig":
        """빠른 확인용 저해상도 설정을 돌려준다(규격 검증 X)."""
        import dataclasses

        return dataclasses.replace(
            self, width=540, height=960, fps=15, preset="ultrafast", video_bitrate="2000k"
        )
