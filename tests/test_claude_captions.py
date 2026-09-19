"""Claude 멀티모달 자막 생성 테스트.

실제 API 는 호출하지 않는다. 가짜 클라이언트로 요청 모양과 실패 대응을 검증한다.
"""

import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from shorts_pipeline.captions import CaptionPlan
from shorts_pipeline.claude_captions import (
    CAPTION_SCHEMA,
    MAX_IMAGE_SIDE,
    CaptionGenerationError,
    ClaudeCaptionGenerator,
    build_messages,
    encode_image,
    generate_with_fallback,
    parse_response,
    scrub_personal_info,
    scrub_plan,
)


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "blurred.jpg"
    rng = np.random.default_rng(0)
    cv2.imwrite(str(p), rng.integers(0, 255, (1500, 2000, 3), dtype=np.uint8))
    return p


class FakeMessages:
    """요청을 기록하고 미리 정한 응답이나 예외를 돌려준다."""

    def __init__(self, payload=None, error=None, stop_reason="end_turn"):
        self.payload = payload
        self.error = error
        self.stop_reason = stop_reason
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        if self.error:
            raise self.error
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload, ensure_ascii=False)
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)],
            stop_reason=self.stop_reason,
        )


def fake_client(payload=None, error=None, stop_reason="end_turn"):
    return SimpleNamespace(messages=FakeMessages(payload, error, stop_reason))


GOOD = {
    "title": "관광 채용트렌드 수업",
    "captions": ["오늘의 주제는 채용트렌드", "모둠별로 자료를 정리", "생각을 나누는 시간"],
    "hashtags": ["#진로수업", "#관광과", "#Shorts"],
}


# ---------- 이미지 인코딩 ----------
def test_encode_image_downscales_for_cost(photo):
    block = encode_image(photo)
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/jpeg"
    raw = base64.standard_b64decode(block["source"]["data"])
    with Image.open(io.BytesIO(raw)) as im:
        assert max(im.size) <= MAX_IMAGE_SIDE


def test_encode_image_respects_custom_side(photo):
    raw = base64.standard_b64decode(encode_image(photo, max_side=256)["source"]["data"])
    with Image.open(io.BytesIO(raw)) as im:
        assert max(im.size) <= 256


def test_build_messages_puts_images_before_text(photo):
    msgs = build_messages("설명", [photo, photo], 18.0)
    content = msgs[0]["content"]
    assert [b["type"] for b in content] == ["image", "image", "text"]
    assert "2장" in content[-1]["text"]


# ---------- 개인정보 후처리 ----------
@pytest.mark.parametrize("text,label", [
    ("한빛고등학교 축제 현장", "학교명"),
    ("3학년 2반 친구들과 함께", "학년/반"),
    ("2학년 진로 수업", "학년"),
])
def test_scrub_removes_identifying_info(text, label):
    cleaned, found = scrub_personal_info(text)
    assert label in found
    assert "고등학교" not in cleaned
    assert "학년" not in cleaned or label == "학교명"


def test_scrub_keeps_clean_text_unchanged():
    text = "모둠별로 자료를 정리하는 중"
    cleaned, found = scrub_personal_info(text)
    assert cleaned == text and found == []


def test_scrub_plan_covers_title_and_captions():
    plan = CaptionPlan("행복고등학교 축제", ["3학년 2반의 하루", "즐거운 시간"], ["#태그"])
    cleaned, flagged = scrub_plan(plan)
    assert "고등학교" not in cleaned.title
    assert "학년" not in cleaned.captions[0]
    assert cleaned.captions[1] == "즐거운 시간"
    assert flagged


def test_scrub_never_empties_a_caption():
    """전부 걸러져도 빈 자막을 남기지 않는다(영상에 빈 칸이 생긴다)."""
    plan = CaptionPlan("제목", ["3학년"], [])
    cleaned, _ = scrub_plan(plan)
    assert cleaned.captions[0]


# ---------- 응답 해석 ----------
def test_parse_response_reads_plan():
    plan = parse_response(json.dumps(GOOD, ensure_ascii=False), 3)
    assert plan.title == GOOD["title"]
    assert len(plan.captions) == 3


def test_parse_response_tolerates_count_mismatch():
    plan = parse_response(json.dumps(GOOD, ensure_ascii=False), 5)
    assert len(plan.fitted(5).captions) == 5


def test_parse_response_rejects_empty_title():
    with pytest.raises(CaptionGenerationError):
        parse_response(json.dumps({"title": "  ", "captions": ["가"]}), 1)


def test_parse_response_rejects_empty_captions():
    with pytest.raises(CaptionGenerationError):
        parse_response(json.dumps({"title": "제목", "captions": []}), 1)


# ---------- 요청 모양 ----------
def test_request_uses_structured_output_and_adaptive_thinking(photo):
    client = fake_client(GOOD)
    gen = ClaudeCaptionGenerator(client=client, effort="medium")
    gen.generate("설명", [photo], 1)
    req = client.messages.last_request
    assert req["model"] == "claude-opus-5"
    assert req["thinking"] == {"type": "adaptive"}
    assert req["output_config"]["effort"] == "medium"
    assert req["output_config"]["format"]["schema"] == CAPTION_SCHEMA
    assert "학생 이름" in req["system"]


def test_explicit_title_overrides_model_output(photo):
    gen = ClaudeCaptionGenerator(client=fake_client(GOOD), title="내 제목")
    assert gen.generate("설명", [photo], 3).title == "내 제목"


def test_overlong_model_title_is_shortened(photo):
    payload = dict(GOOD, title="아주 긴 제목을 모델이 돌려준 경우를 대비한 문장입니다")
    plan = ClaudeCaptionGenerator(client=fake_client(payload)).generate("설명", [photo], 3)
    assert len(plan.title) <= 25


# ---------- 실패 처리 ----------
def test_no_images_raises(tmp_path):
    with pytest.raises(CaptionGenerationError):
        ClaudeCaptionGenerator(client=fake_client(GOOD)).generate("설명", [], 0)


def test_refusal_raises(photo):
    gen = ClaudeCaptionGenerator(client=fake_client(GOOD, stop_reason="refusal"))
    with pytest.raises(CaptionGenerationError):
        gen.generate("설명", [photo], 3)


def test_malformed_json_raises(photo):
    gen = ClaudeCaptionGenerator(client=fake_client("이건 JSON 이 아닙니다"))
    with pytest.raises(CaptionGenerationError):
        gen.generate("설명", [photo], 3)


def test_api_error_becomes_caption_error(photo):
    import anthropic

    err = anthropic.APIConnectionError(request=SimpleNamespace())
    gen = ClaudeCaptionGenerator(client=fake_client(error=err))
    with pytest.raises(CaptionGenerationError):
        gen.generate("설명", [photo], 3)


# ---------- 대체 동작 ----------
def test_fallback_returns_template_plan_on_failure(photo):
    gen = ClaudeCaptionGenerator(client=fake_client("깨진 응답"))
    plan, warnings = generate_with_fallback("과학 실험 수업", [photo], 3, generator=gen)
    assert len(plan.captions) == 3
    assert plan.captions[0] == "과학 실험 수업"  # 템플릿 생성기의 첫 자막
    assert any("실패" in w for w in warnings)


def test_fallback_passes_through_on_success(photo):
    gen = ClaudeCaptionGenerator(client=fake_client(GOOD))
    plan, warnings = generate_with_fallback("설명", [photo], 3, generator=gen)
    assert plan.title == GOOD["title"]
    assert any("사실관계" in w for w in warnings)


def test_fallback_warns_when_pii_was_scrubbed(photo):
    payload = dict(GOOD, title="빛나고등학교 진로수업")
    gen = ClaudeCaptionGenerator(client=fake_client(payload))
    plan, warnings = generate_with_fallback("설명", [photo], 3, generator=gen)
    assert "고등학교" not in plan.title
    assert any("식별 정보" in w for w in warnings)


# ---------- 자격 증명 (실제로 났던 버그) ----------
def test_missing_credentials_raises_caption_error_not_typeerror(monkeypatch, photo):
    """SDK 는 요청 시점에 TypeError 를 던진다. 그대로 두면 렌더링까지 죽었다."""
    from shorts_pipeline import claude_captions

    for var in claude_captions.CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(claude_captions, "CREDENTIAL_PROFILE_DIR", Path("/nonexistent-profile-dir"))

    gen = ClaudeCaptionGenerator()  # 가짜 클라이언트 없이 = 실제 경로
    with pytest.raises(CaptionGenerationError, match="자격 증명"):
        gen.generate("설명", [photo], 1)


def test_missing_credentials_still_produces_captions_via_fallback(monkeypatch, photo):
    from shorts_pipeline import claude_captions

    for var in claude_captions.CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(claude_captions, "CREDENTIAL_PROFILE_DIR", Path("/nonexistent-profile-dir"))

    plan, warnings = generate_with_fallback("미술 수업", [photo], 3)
    assert len(plan.captions) == 3
    assert any("자격 증명" in w for w in warnings)


def test_has_credentials_sees_env_var(monkeypatch):
    from shorts_pipeline import claude_captions

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert claude_captions.has_credentials()


def test_has_credentials_sees_cli_profile(monkeypatch, tmp_path):
    from shorts_pipeline import claude_captions

    for var in claude_captions.CREDENTIAL_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    profile = tmp_path / "anthropic"
    profile.mkdir()
    (profile / "profile.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(claude_captions, "CREDENTIAL_PROFILE_DIR", profile)
    assert claude_captions.has_credentials()


def test_unexpected_exception_is_wrapped(photo):
    """예상 못 한 오류로 영상 생성까지 막히면 안 된다."""
    gen = ClaudeCaptionGenerator(client=fake_client(error=RuntimeError("갑작스런 오류")))
    with pytest.raises(CaptionGenerationError, match="예상하지 못한"):
        gen.generate("설명", [photo], 1)
