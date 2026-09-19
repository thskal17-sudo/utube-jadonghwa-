"""Phase 2: Claude 멀티모달로 사진에 맞는 제목·자막 생성.

Phase 1 의 TemplateCaptionGenerator 는 고정 문구를 돌려썼다. 과학 실험이든
진로 수업이든 "오늘도 한 뼘 더 성장" 같은 자막이 붙었다. 이 모듈은 같은
CaptionPlan 을 Claude 가 사진을 직접 보고 채우게 한다.

개인정보 관련 설계
- **블러 처리된 사진만 보낸다.** 파이프라인이 블러 단계를 마친 뒤 그 결과물을
  넘긴다. 원본이 외부로 나가지 않는다. generate() 는 이를 인자로 강제한다.
- 전송 전에 사진을 축소한다. 화질이 아니라 내용 파악이 목적이고, 이미지 토큰이
  비용의 대부분이기 때문이다.
- 응답에서 학교명·학년·반 같은 식별 정보를 후처리로 걸러낸다. 프롬프트에도
  금지를 적지만, 생성형이라 규칙만으로는 보장되지 않는다.

비용
사진 6장을 긴 변 768px 로 줄여 보내면 이미지가 약 6천 토큰, 응답이 수백 토큰
수준이다. Claude Opus 5 기준 영상 한 편당 대략 $0.05 안팎이다.

실패하면 예외를 던지지 않고 None 을 돌려준다. 파이프라인이 템플릿 생성기로
안전하게 되돌아가도록 하기 위해서다. 자막이 밋밋한 것은 불편이지만, 영상이
아예 안 나오는 것은 실패다.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image, ImageOps

from .captions import CaptionPlan, TemplateCaptionGenerator, make_title

log = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "medium"
MAX_IMAGE_SIDE = 768       # 내용 파악에 충분하면서 토큰을 아끼는 크기
JPEG_QUALITY = 80
MAX_TOKENS = 16000

CAPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "18자 이내의 숏폼 제목"},
        "captions": {
            "type": "array",
            "items": {"type": "string", "description": "25자 이내의 장면 자막"},
            "description": "사진 순서와 1:1 로 대응하는 자막 목록",
        },
        "hashtags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "captions", "hashtags"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """당신은 교사가 학교 활동 사진으로 유튜브 숏폼을 만드는 것을 돕습니다.
사진은 개인정보 보호를 위해 얼굴이 이미 가려진 상태입니다. 가려진 부분을 추측하지 마세요.

반드시 지킬 것:
- 학교명, 학년, 반, 학생 이름, 교사 이름, 지역명을 절대 쓰지 마세요.
- 사진에서 확실히 보이는 것만 쓰세요. 추측을 단정적으로 쓰지 마세요.
- 과장된 낚시성 표현을 쓰지 마세요. 교육 기관의 공식 계정에 올라갑니다.
- 이모지를 쓰지 마세요. 자막 이미지에서 깨집니다."""

USER_TEMPLATE = """활동 사진 {n}장입니다. 순서대로 첨부했습니다.
교사가 적은 활동 설명: "{description}"

이 사진들로 {duration:.0f}초짜리 세로 숏폼을 만듭니다. 제목과 장면별 자막을 써 주세요.

- 제목: 18자 이내. 내용이 한눈에 들어오고 눌러보고 싶은 문구.
- 자막: 정확히 {n}개. 각 사진에 1:1 로 대응. 한 개당 25자 이내.
- 첫 자막은 주제를 알리고, 마지막 자막은 마무리 느낌으로.
- 해시태그: 3~5개.

각 사진에서 실제로 무엇을 하고 있는지 보고, 활동 설명과 맞물리게 써 주세요."""

# 자막에 남으면 안 되는 식별 정보
BANNED_PATTERNS = [
    (re.compile(r"[가-힣]{1,10}(초등학교|중학교|고등학교|초교|중교|고교)"), "학교명"),
    (re.compile(r"\d+\s*학년\s*\d+\s*반"), "학년/반"),
    (re.compile(r"\d+\s*학년"), "학년"),
    (re.compile(r"\d+\s*반\b"), "반"),
]


CREDENTIAL_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
CREDENTIAL_PROFILE_DIR = Path.home() / ".config" / "anthropic"


def has_credentials() -> bool:
    """Claude 를 호출할 자격 증명이 있는지 확인한다.

    SDK 는 환경변수 외에 `ant auth login` 프로필도 읽는다. 환경변수만 보고
    없다고 단정하면 프로필을 쓰는 사용자를 막게 된다.
    """
    if any(os.environ.get(v) for v in CREDENTIAL_ENV_VARS):
        return True
    return CREDENTIAL_PROFILE_DIR.is_dir() and any(CREDENTIAL_PROFILE_DIR.iterdir())


class CaptionGenerationError(Exception):
    """Claude 자막 생성이 실패했을 때. 파이프라인은 템플릿으로 대체한다."""


def encode_image(path: Path, max_side: int = MAX_IMAGE_SIDE) -> dict:
    """사진을 축소해 base64 이미지 블록으로 만든다."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buf.getvalue()).decode("ascii"),
        },
    }


def scrub_personal_info(text: str) -> tuple:
    """식별 정보를 지운다. (정리된 문구, 걸린 항목들) 을 돌려준다."""
    found: List[str] = []
    out = text
    for pattern, label in BANNED_PATTERNS:
        if pattern.search(out):
            found.append(label)
            out = pattern.sub("", out)
    return " ".join(out.split()).strip(" ,.·-"), found


def scrub_plan(plan: CaptionPlan) -> tuple:
    """계획 전체에서 식별 정보를 걸러낸다."""
    flagged: List[str] = []
    title, hits = scrub_personal_info(plan.title)
    flagged += hits
    captions = []
    for c in plan.captions:
        cleaned, hits = scrub_personal_info(c)
        flagged += hits
        captions.append(cleaned or c)
    return CaptionPlan(title or plan.title, captions, list(plan.hashtags)), sorted(set(flagged))


def build_messages(description: str, image_paths: Sequence[Path], duration: float) -> list:
    content: List[dict] = [encode_image(p) for p in image_paths]
    content.append({
        "type": "text",
        "text": USER_TEMPLATE.format(
            n=len(image_paths), description=" ".join(description.split()), duration=duration
        ),
    })
    return [{"role": "user", "content": content}]


def parse_response(raw: str, n_scenes: int) -> CaptionPlan:
    data = json.loads(raw)
    title = str(data["title"]).strip()
    captions = [str(c).strip() for c in data.get("captions", []) if str(c).strip()]
    if not title:
        raise CaptionGenerationError("제목이 비어 있습니다.")
    if not captions:
        raise CaptionGenerationError("자막이 비어 있습니다.")
    if len(captions) != n_scenes:
        # 개수가 어긋나도 버리지 않는다. fitted() 가 장면 수에 맞춰 준다.
        log.warning("[claude] 자막 %d개를 받았습니다(사진 %d장). 개수를 맞춥니다.",
                    len(captions), n_scenes)
    hashtags = [str(h).strip() for h in data.get("hashtags", []) if str(h).strip()]
    return CaptionPlan(title=title, captions=captions, hashtags=hashtags)


class ClaudeCaptionGenerator:
    """사진을 보고 제목·자막을 쓰는 생성기.

    generate() 에는 반드시 **블러 처리된** 사진 경로를 넘길 것.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        title: Optional[str] = None,
        duration: float = 18.0,
        max_image_side: int = MAX_IMAGE_SIDE,
        client=None,
    ):
        self.model = model
        self.effort = effort
        self.title = title
        self.duration = duration
        self.max_image_side = max_image_side
        self._client = client
        self.last_flagged: List[str] = []

    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:
            raise CaptionGenerationError(
                "anthropic 패키지가 없습니다. `pip install anthropic` 후 다시 실행하세요."
            ) from e
        if not has_credentials():
            # SDK 는 클라이언트를 만들 때가 아니라 요청을 보낼 때 TypeError 를 던진다.
            # 그대로 두면 파이프라인이 죽으므로 여기서 미리 잡아 안내한다.
            raise CaptionGenerationError(
                "Claude API 자격 증명이 없습니다. ANTHROPIC_API_KEY 환경변수를 설정하세요. "
                "(https://console.anthropic.com 에서 발급)"
            )
        try:
            self._client = anthropic.Anthropic()
        except Exception as e:
            raise CaptionGenerationError(f"Claude 클라이언트를 만들지 못했습니다: {e}") from e
        return self._client

    def generate(self, description: str, image_paths: Sequence[Path], n_scenes: int) -> CaptionPlan:
        """사진과 설명으로 자막 계획을 만든다. 실패하면 CaptionGenerationError."""
        import anthropic

        if not image_paths:
            raise CaptionGenerationError("사진이 없습니다.")
        client = self._get_client()
        messages = build_messages(description, image_paths, self.duration)

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": CAPTION_SCHEMA},
                },
            )
        except anthropic.AuthenticationError as e:
            raise CaptionGenerationError(
                "Claude 인증에 실패했습니다. ANTHROPIC_API_KEY 를 설정하거나 `ant auth login` 하세요."
            ) from e
        except anthropic.RateLimitError as e:
            raise CaptionGenerationError(f"요청이 제한됐습니다. 잠시 뒤 다시 시도하세요: {e}") from e
        except anthropic.APIStatusError as e:
            raise CaptionGenerationError(f"Claude API 오류({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise CaptionGenerationError(f"Claude 에 연결하지 못했습니다: {e}") from e
        except CaptionGenerationError:
            raise
        except Exception as e:
            # 자막은 있으면 좋은 것이고 영상은 나와야 하는 것이다. 예상 못 한
            # 오류로 렌더링까지 막지 않는다. (예: 자격 증명 미해결 TypeError)
            raise CaptionGenerationError(f"예상하지 못한 오류({type(e).__name__}): {e}") from e

        if getattr(response, "stop_reason", None) == "refusal":
            raise CaptionGenerationError("Claude 가 이 요청에 대한 응답을 거절했습니다.")

        text = next((b.text for b in response.content if b.type == "text"), None)
        if not text:
            raise CaptionGenerationError("응답에 텍스트가 없습니다.")

        try:
            plan = parse_response(text, n_scenes)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise CaptionGenerationError(f"응답을 해석하지 못했습니다: {e}") from e

        plan, flagged = scrub_plan(plan)
        self.last_flagged = flagged
        if flagged:
            log.warning("[claude] 자막에서 식별 정보를 걸러냈습니다: %s", ", ".join(flagged))
        if self.title:
            plan = CaptionPlan(self.title, plan.captions, plan.hashtags)
        elif len(plan.title) > 24:
            plan = CaptionPlan(make_title(plan.title), plan.captions, plan.hashtags)
        return plan


def generate_with_fallback(
    description: str,
    image_paths: Sequence[Path],
    n_scenes: int,
    generator: Optional[ClaudeCaptionGenerator] = None,
    seed: int = 0,
    title: Optional[str] = None,
) -> tuple:
    """Claude 로 자막을 만들되, 실패하면 템플릿으로 되돌아간다.

    (계획, 경고 목록) 을 돌려준다. 자막이 밋밋한 것은 불편이지만 영상이
    아예 안 나오는 것은 실패이므로, 여기서 예외를 삼킨다.
    """
    generator = generator or ClaudeCaptionGenerator(title=title)
    warnings: List[str] = []
    try:
        plan = generator.generate(description, image_paths, n_scenes)
        if generator.last_flagged:
            warnings.append(
                f"자동 생성 자막에서 식별 정보를 걸러냈습니다({', '.join(generator.last_flagged)}). "
                "최종 자막을 한 번 읽어 보세요."
            )
        warnings.append("자막을 Claude 가 생성했습니다. 사실관계를 사람이 확인하세요.")
        return plan, warnings
    except CaptionGenerationError as e:
        log.warning("[claude] 자막 생성 실패 → 템플릿으로 대체합니다: %s", e)
        warnings.append(f"Claude 자막 생성에 실패해 기본 템플릿을 썼습니다: {e}")
        plan = TemplateCaptionGenerator(title=title, seed=seed).generate(
            description, image_paths, n_scenes
        )
        return plan, warnings
