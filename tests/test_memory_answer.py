from datetime import datetime

from app.components.memory_answer import (
    NO_RESULT_TEXT,
    SUGGESTION_HEADER,
    Suggestion,
    compose_answer,
    template_headline,
    unknown_place_notices,
)
from tests.memory_fakes import make_record

# ---- 앞머리 문장 (LLM이 못 쓸 때의 템플릿) -----------------------------------


def test_headline_without_any_result_is_the_no_result_text():
    assert template_headline(["가을"], found=0) == NO_RESULT_TEXT
    assert template_headline([], found=0) == NO_RESULT_TEXT


def test_headline_lists_the_conditions_and_the_count():
    assert template_headline(["홍대 근처", "비 오는 날"], found=2) == (
        "홍대 근처, 비 오는 날 조건에 맞는 자물쇠 2건을 찾았어요."
    )


def test_headline_without_conditions_says_similar():
    assert template_headline([], found=4) == "비슷한 자물쇠 4건을 찾았어요."


def test_headline_when_only_part_of_the_matches_is_sent_says_the_total():
    assert template_headline(["가을"], found=50, total=120) == (
        "가을 조건에 맞는 자물쇠는 총 120건이에요. 그중 상위 50건을 보여드려요."
    )


def test_headline_total_equal_to_found_is_just_the_count():
    assert (
        template_headline(["가을"], found=3, total=3) == "가을 조건에 맞는 자물쇠 3건을 찾았어요."
    )


def test_headline_with_only_suggestions_says_nothing_matches_all_conditions():
    assert template_headline(["가을", "서울숲 근처"], found=0, suggestions=2) == (
        "가을, 서울숲 근처 조건에 모두 맞는 자물쇠는 없어요."
    )


def test_headline_never_lists_records():
    # 목록은 프론트가 matchedRecords로 카드를 보여준다 — 문장에는 값(곡·날짜 등)이 없다
    headline = template_headline(["가을"], found=3)

    assert "\n" not in headline


# ---- 문장 합치기: 앞머리 + 제안 + 안내 ---------------------------------------


def _suggestions():
    return [
        Suggestion(
            make_record(
                2,
                created_at=datetime(2026, 5, 2, 9, 0),
                place_name="성수동2가",
                title="Blueming",
                artist_name="아이유",
            ),
            ("서울숲 근처",),
        ),
        Suggestion(
            make_record(
                3,
                created_at=datetime(2025, 9, 14, 9, 0),
                place_name="반포동",
                title="두 사람",
                artist_name="성시경",
            ),
            ("가을",),
        ),
    ]


def test_answer_with_only_a_headline_is_the_headline():
    assert compose_answer("가을 조건에 맞는 자물쇠 3건을 찾았어요.") == (
        "가을 조건에 맞는 자물쇠 3건을 찾았어요."
    )


def test_suggestions_follow_the_headline_after_a_blank_line():
    answer = compose_answer("가을 조건에 맞는 자물쇠 1건을 찾았어요.", _suggestions())

    assert answer == (
        "가을 조건에 맞는 자물쇠 1건을 찾았어요.\n"
        "\n"
        f"{SUGGESTION_HEADER}\n"
        "1. 2026년 5월 2일 · 성수동2가 · 아이유 - Blueming ('서울숲 근처'만 맞아요)\n"
        "2. 2025년 9월 14일 · 반포동 · 성시경 - 두 사람 ('가을'만 맞아요)"
    )


def test_suggestion_lists_every_matched_condition():
    suggestion = Suggestion(make_record(2), ("가을", "서울숲 근처"))

    assert compose_answer("앞머리", [suggestion]).endswith("('가을', '서울숲 근처'만 맞아요)")


def test_missing_place_is_left_out_of_a_suggestion_instead_of_printing_none():
    answer = compose_answer("앞머리", [Suggestion(make_record(2, place_name=None), ("가을",))])

    assert "None" not in answer
    assert answer.splitlines()[-1] == "1. 2026년 8월 24일 · 아이유 - 밤편지 ('가을'만 맞아요)"


def test_suggestion_without_matched_conditions_has_no_note():
    answer = compose_answer("앞머리", [Suggestion(make_record(2), ())])

    assert answer.splitlines()[-1] == "1. 2026년 8월 24일 · 오산동 · 아이유 - 밤편지"


# ---- 안내 문장 ---------------------------------------------------------------


def test_notices_come_after_everything_separated_by_a_blank_line():
    answer = compose_answer("앞머리", _suggestions(), notices=["안내 하나", "안내 둘"])

    assert answer.splitlines()[-3:] == ["", "안내 하나", "안내 둘"]
    assert answer.startswith("앞머리\n\n" + SUGGESTION_HEADER)


def test_notices_are_kept_even_when_nothing_is_found():
    assert compose_answer(NO_RESULT_TEXT, notices=["안내 하나"]) == f"{NO_RESULT_TEXT}\n\n안내 하나"


def test_no_notices_leaves_the_answer_unchanged():
    assert compose_answer("앞머리", _suggestions(), notices=[]) == compose_answer(
        "앞머리", _suggestions()
    )


def test_unknown_place_found_in_comments_is_explained():
    assert unknown_place_notices(found=["광교산"], missing=[]) == [
        "'광교산'은 장소 목록에 없는 곳이라서, 기록에 적어 둔 코멘트에서 찾았어요."
    ]


def test_unknown_place_not_found_says_the_place_condition_was_not_applied_and_how_to_ask():
    assert unknown_place_notices(found=[], missing=["북한산"]) == [
        "'북한산'은 아직 모르는 장소라서 위치 조건은 적용하지 못했어요. "
        "동이나 구 이름(예: 성수동, 성동구)으로 물어보시면 더 정확하게 찾을 수 있어요."
    ]


def test_found_and_missing_places_get_separate_sentences_and_the_hint_comes_once():
    notices = unknown_place_notices(found=["광교산"], missing=["북한산", "관악산"])

    assert len(notices) == 2
    assert notices[0].startswith("'광교산'은 장소 목록에 없는")
    assert notices[1].startswith("'북한산', '관악산'은 아직 모르는 장소")
    assert sum("성수동" in n for n in notices) == 1


def test_topic_particle_follows_the_last_letter():
    with_batchim = unknown_place_notices(found=["광교산"], missing=[])[0]
    without_batchim = unknown_place_notices(found=["롯데타워"], missing=[])[0]
    not_hangul = unknown_place_notices(found=["Everland"], missing=[])[0]

    assert with_batchim.startswith("'광교산'은 ")
    assert without_batchim.startswith("'롯데타워'는 ")
    assert not_hangul.startswith("'Everland'은(는) ")


def test_no_places_means_no_notices():
    assert unknown_place_notices(found=[], missing=[]) == []
