"""2·3단계: 활동 설명 → 제목/장면별 자막 계획.

Phase 1(MVP)은 규칙 기반 템플릿(TemplateCaptionGenerator)으로 채운다.
Phase 2에서는 같은 CaptionPlan 형식을 Claude 멀티모달이 생성하도록
교체하면 되고, --captions 옵션으로 JSON을 직접 넘길 수도 있다.

규칙: 학교명·학년·반·학생 이름 등 식별 정보는 자막에 넣지 않는다.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Protocol, Sequence


@dataclass
class CaptionPlan:
    title: str
    captions: List[str]  # 장면(사진) 순서대로 하나씩
    hashtags: List[str] = field(default_factory=list)

    def fitted(self, n_scenes: int) -> "CaptionPlan":
        """장면 수에 맞게 자막 목록을 늘리거나 줄인다."""
        if not self.captions:
            caps = [self.title] * n_scenes
        else:
            caps = [self.captions[i % len(self.captions)] for i in range(n_scenes)]
        return CaptionPlan(self.title, caps, list(self.hashtags))

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "CaptionPlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            title=str(data["title"]),
            captions=[str(c) for c in data.get("captions", [])],
            hashtags=[str(h) for h in data.get("hashtags", [])],
        )


class CaptionGenerator(Protocol):
    def generate(self, description: str, image_paths: Sequence[Path], n_scenes: int) -> CaptionPlan: ...


MIDDLE_TEMPLATES = [
    "집중하는 눈빛이 반짝반짝",
    "직접 해보니까 더 재밌다!",
    "함께라서 더 즐거운 시간",
    "하나하나 손으로 만들어 가는 중",
    "오늘도 한 뼘 더 성장",
    "친구들과 머리를 맞대고",
    "이 순간, 교실이 가장 뜨거울 때",
    "작은 도전이 모여 큰 배움으로",
]

CLOSING_TEMPLATES = [
    "다음 활동도 기대해 주세요!",
    "우리 반의 오늘, 여기까지!",
    "이런 수업이라면 매일 하고 싶다",
]

DEFAULT_HASHTAGS = ["#학교생활", "#교실이야기", "#Shorts"]


def make_title(description: str, max_len: int = 18) -> str:
    """설명 문장을 제목 길이로 다듬는다(숏폼 상단 제목용)."""
    text = " ".join(description.split()).rstrip(".!?~ ")
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    if " " in cut[8:]:
        cut = cut[: cut.rfind(" ")]
    return cut.rstrip(",. ") + "…"


class TemplateCaptionGenerator:
    """Phase 1: 사용자 설명 + 고정 템플릿 조합."""

    def __init__(self, title: Optional[str] = None, seed: int = 0):
        self.title = title
        self.seed = seed

    def generate(self, description: str, image_paths: Sequence[Path], n_scenes: int) -> CaptionPlan:
        rng = random.Random(self.seed)
        description = " ".join(description.split())
        title = self.title or make_title(description)

        captions: List[str] = [description]
        middles = MIDDLE_TEMPLATES[:]
        rng.shuffle(middles)
        for i in range(1, max(1, n_scenes - 1)):
            captions.append(middles[(i - 1) % len(middles)])
        if n_scenes >= 2:
            captions.append(rng.choice(CLOSING_TEMPLATES))
        return CaptionPlan(title=title, captions=captions[:n_scenes], hashtags=list(DEFAULT_HASHTAGS))
