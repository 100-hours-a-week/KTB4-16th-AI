from datetime import datetime

from app.components.memory_answer import NO_RESULT_TEXT, compose_answer
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
