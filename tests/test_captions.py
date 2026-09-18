import pytest

from shorts_pipeline.captions import CaptionPlan, TemplateCaptionGenerator, make_title


def test_make_title_short_passthrough():
    assert make_title("가을 운동회") == "가을 운동회"


def test_make_title_truncates_long_text():
    title = make_title("3학년 2반 과학 시간에 화산 폭발 실험을 진행했습니다", max_len=18)
    assert len(title) <= 19  # 말줄임표 포함
    assert title.endswith("…")


def test_generator_produces_one_caption_per_scene():
    plan = TemplateCaptionGenerator(seed=0).generate("미술 시간 찰흙 공예", [], 5)
    assert len(plan.captions) == 5
    assert plan.captions[0] == "미술 시간 찰흙 공예"
    assert plan.title


def test_generator_is_deterministic():
    a = TemplateCaptionGenerator(seed=3).generate("체육대회", [], 6)
    b = TemplateCaptionGenerator(seed=3).generate("체육대회", [], 6)
    assert a.captions == b.captions


def test_explicit_title_wins():
    plan = TemplateCaptionGenerator(title="나의 제목", seed=0).generate("아주 긴 설명입니다", [], 3)
    assert plan.title == "나의 제목"


@pytest.mark.parametrize("n", [1, 2, 3, 8, 12])
def test_fitted_matches_scene_count(n):
    plan = CaptionPlan("제목", ["하나", "둘", "셋"]).fitted(n)
    assert len(plan.captions) == n


def test_fitted_with_empty_captions_uses_title():
    plan = CaptionPlan("제목만", []).fitted(3)
    assert plan.captions == ["제목만"] * 3


def test_caption_plan_json_roundtrip(tmp_path):
    plan = CaptionPlan("제목", ["가", "나"], ["#태그"])
    path = tmp_path / "c.json"
    plan.to_json(path)
    loaded = CaptionPlan.from_json(path)
    assert (loaded.title, loaded.captions, loaded.hashtags) == (plan.title, plan.captions, plan.hashtags)
