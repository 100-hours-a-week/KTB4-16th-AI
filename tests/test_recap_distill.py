"""RECAP 지식 증류 도구(scripts/recap_distill) — 가짜 입력 합성, 자동 채점, 비용 계산.

외부 API는 부르지 않는다. 합성 입력은 서비스 코드로 만들기 때문에, 서비스 프롬프트 형식이
바뀌면 이 시험이 먼저 알려 준다.
"""

from scripts.recap_distill import pools
from scripts.recap_distill.generate_labels import PRICES, cost_of, estimate_cost
from scripts.recap_distill.rubric import check
from scripts.recap_distill.synth_inputs import build_one

PROMPT = (
    "[이달의 통계]\n가장 많이 들은 아티스트 아이유, 가장 많이 간 장소 성수동\n\n"
    "[이달의 기록]\n1. 곡 분위기: 잔잔한 발라드 / 아티스트: 아이유 / 장소: 성수동 / "
    "코멘트: 친구랑 수다 떨다 보니 시간이 훅\n"
)
GOOD = "당신은 이번 달 주로 성수동에서 아이유의 잔잔한 곡을 들었어요."


# ---- 자동 채점 ----


def test_good_sentence_passes():
    assert check(GOOD, PROMPT) == []


def test_must_start_with_you_and_fit_the_backend_column():
    assert "'당신은'으로 시작하지 않음" in check("이번 달은 성수동이 많았어요.", PROMPT)
    assert "100자 초과" in check("당신은 " + "가" * 100 + "했어요.", PROMPT)


def test_two_sentences_or_a_line_break_is_not_one_sentence():
    assert "한 문장이 아님" in check("당신은 성수동에 갔어요. 아이유를 들었어요.", PROMPT)
    assert "한 문장이 아님" in check("당신은 성수동에 갔어요\n아이유를 들었어요", PROMPT)


def test_mood_score_must_not_be_mentioned():
    assert "기분점수를 언급함" in check("당신은 기분점수가 높은 달이었어요.", PROMPT)
    assert "기분점수를 언급함" in check("당신은 평균 32점으로 지냈어요.", PROMPT)


def test_names_not_in_the_input_are_invented():
    violations = check("당신은 이번 달 연남동에서 Adele를 들었어요.", PROMPT)

    assert any(v.startswith("입력에 없는 이름") for v in violations)


def test_relationship_words_are_allowed_only_when_the_input_has_them():
    """코멘트에 '친구'가 있으면 말해도 되고(입력에 있음), 없는데 말하면 추측이다."""
    assert check("당신은 친구들과 성수동에 갔어요.", PROMPT) == []
    no_friend = PROMPT.replace("친구랑", "혼자")

    assert any(
        v.startswith("입력에 없는 관계") for v in check("당신은 연인과 성수동에 갔어요.", no_friend)
    )


def test_comment_must_not_be_quoted_verbatim():
    quoted = "당신은 '친구랑 수다 떨다 보니 시간이 훅' 갔던 성수동을 좋아했어요."

    assert "코멘트를 그대로 인용함" in check(quoted, PROMPT)


def test_empty_text_fails():
    assert check("  ", PROMPT) == ["비어 있음"]


# ---- 가짜 입력 합성 ----


def test_synthetic_input_is_deterministic_for_a_seed():
    assert build_one(7, seed=1) == build_one(7, seed=1)
    assert build_one(7, seed=1) != build_one(8, seed=1)


def test_synthetic_input_has_the_same_sections_as_the_service_prompt():
    row = build_one(3, seed=20261016)

    assert row["prompt"].startswith("[이달의 통계]")
    assert "[이달의 기록]\n1. " in row["prompt"]
    assert "[이달의 사진 분위기 비중]" in row["prompt"]


def test_synthetic_data_only_uses_pool_names():
    """채점이 '입력에 없는 이름'을 가리려면 입력의 아티스트·장소가 재료 목록 안에 있어야 한다."""
    for i in range(20):
        prompt = build_one(i, seed=5)["prompt"]
        head = prompt.splitlines()[1]
        artist = head.split("아티스트 ")[1].split(",")[0]

        assert artist in pools.ARTISTS


def test_snow_never_comes_with_warm_temperature():
    import re

    for i in range(60):
        for line in build_one(i, seed=9)["prompt"].splitlines():
            if match := re.search(r"날씨: (?:눈|진눈깨비) (-?[\d.]+)도", line):
                assert float(match.group(1)) <= 1.5


# ---- 비용 계산 ----


def test_cost_uses_the_published_per_million_token_prices():
    price_in, price_out = PRICES["claude-haiku-5-5"]

    assert cost_of("claude-haiku-5-5", 1_000_000, 0) == price_in
    assert cost_of("claude-haiku-5-5", 0, 1_000_000) == price_out


def test_estimate_scales_with_count_and_never_runs_for_unknown_models():
    one = estimate_cost("claude-haiku-5-5", ["가" * 1000])

    assert estimate_cost("claude-haiku-5-5", ["가" * 1000] * 10) > one * 9.9
    assert "claude-unknown" not in PRICES
