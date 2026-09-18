"""교육사진 자동 숏폼 제작 파이프라인 (Phase 1 MVP).

사진 폴더 + 활동 설명 → 얼굴 블러 → 켄 번즈 편집 → 자막 → 합성 BGM → 9:16 mp4
"""

from .config import ShortsConfig
from .pipeline import run_pipeline, PipelineReport

__all__ = ["ShortsConfig", "run_pipeline", "PipelineReport"]
__version__ = "0.1.0"
