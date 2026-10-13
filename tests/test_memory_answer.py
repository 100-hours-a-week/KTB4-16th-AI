from datetime import datetime

from app.components.memory_answer import (
    NO_RESULT_TEXT,
    Suggestion,
    compose_answer,
    unknown_place_notices,
)
from tests.memory_fakes import make_record


def test_no_records_says_there_is_nothing():
    assert compose_answer([]) == NO_RESULT_TEXT


def test_answer_lists_only_values_of_the_found_records():
    records = [
        make_record(1, created_at=datetime(2026, 8, 24, 18, 30), place_name="오산동"),
        make_record(2, created_at=datetime(2026, 1, 3, 9, 0), title="Blueming", artist_name="IU"),
    ]

    assert compose_answer(records) == (
        "비슷한 기록 2개를 찾았어요.\n"
        "1. 2026년 8월 24일 · 오산동 · 아이유 - 밤편지\n"
        "2. 2026년 1월 3일 · 오산동 · IU - Blueming"
    )


def test_missing_place_is_left_out_instead_of_printing_none():
    answer = compose_answer([make_record(1, place_name=None)])

    assert "None" not in answer
    assert answer.splitlines()[1] == "1. 2026년 8월 24일 · 아이유 - 밤편지"


# ---- 조건이 있는 질문: 모두 맞는 기록 + 제안 -------------------------------------


def _exact():
    return [make_record(1, created_at=datetime(2026, 10, 3, 9, 0), place_name="성수동1가")]


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


def test_conditioned_answer_with_exact_records_and_suggestions():
    assert compose_answer(_exact(), _suggestions(), conditioned=True) == (
        "조건에 모두 맞는 기록 1개를 찾았어요.\n"
        "1. 2026년 10월 3일 · 성수동1가 · 아이유 - 밤편지\n"
        "\n"
        "이런 기록을 찾으시는 건 아닌가요?\n"
        "1. 2026년 5월 2일 · 성수동2가 · 아이유 - Blueming ('서울숲 근처'만 맞아요)\n"
        "2. 2025년 9월 14일 · 반포동 · 성시경 - 두 사람 ('가을'만 맞아요)"
    )


def test_conditioned_answer_with_only_suggestions_says_no_exact_match():
    assert compose_answer([], _suggestions(), conditioned=True) == (
        "조건에 모두 맞는 기록은 없어요. 이런 기록을 찾으시는 건 아닌가요?\n"
        "1. 2026년 5월 2일 · 성수동2가 · 아이유 - Blueming ('서울숲 근처'만 맞아요)\n"
        "2. 2025년 9월 14일 · 반포동 · 성시경 - 두 사람 ('가을'만 맞아요)"
    )


def test_conditioned_answer_with_only_exact_records_has_no_suggestion_section():
    answer = compose_answer(_exact(), [], conditioned=True)

    assert answer == (
        "조건에 모두 맞는 기록 1개를 찾았어요.\n1. 2026년 10월 3일 · 성수동1가 · 아이유 - 밤편지"
    )


def test_conditioned_answer_with_nothing_is_the_no_result_text():
    assert compose_answer([], [], conditioned=True) == NO_RESULT_TEXT


def test_suggestion_lists_every_matched_condition():
    suggestion = Suggestion(make_record(2), ("가을", "서울숲 근처"))

    answer = compose_answer([], [suggestion], conditioned=True)

    assert answer.endswith("('가을', '서울숲 근처'만 맞아요)")


def test_unconditioned_answer_is_unchanged_by_the_new_parameters():
    assert compose_answer([make_record(1)]).startswith("비슷한 기록 1개를 찾았어요.")


# ---- 안내 문장 ---------------------------------------------------------------


def test_notices_come_after_the_results_separated_by_a_blank_line():
    answer = compose_answer(_exact(), conditioned=True, notices=["안내 하나", "안내 둘"])

    assert answer.splitlines()[-3:] == ["", "안내 하나", "안내 둘"]
    assert answer.startswith("조건에 모두 맞는 기록 1개를 찾았어요.")


def test_notices_are_kept_even_when_nothing_is_found():
    answer = compose_answer([], notices=["안내 하나"])

    assert answer == f"{NO_RESULT_TEXT}\n\n안내 하나"


def test_no_notices_leaves_the_answer_unchanged():
    assert compose_answer(_exact(), conditioned=True, notices=[]) == compose_answer(
        _exact(), conditioned=True
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
