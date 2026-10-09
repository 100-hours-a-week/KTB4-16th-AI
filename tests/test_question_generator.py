from types import SimpleNamespace

from app.components.question_generator import QuestionGenerator, parse
from app.db.repositories.balance_stats_repository import GenreStats, aggregate
from tests.test_song_curator import ScriptedLLM

RESPONSE = """1. 비 오는 날 인디는? | 잔나비 | 검정치마
- 인디는 어디서 들어야 제맛 | 한강 산책 | 방에서 혼자
노래방에서 부르기 좋은 건? | 발라드 | 발라드
형식이 틀린 줄
<질문> | <선택지 A> | <선택지 B>
비 오는 날 인디는? | 잔나비 | 검정치마"""


def test_parse_keeps_only_valid_lines():
    assert parse(RESPONSE) == [
        ("비 오는 날 인디는?", "잔나비", "검정치마"),
        ("인디는 어디서 들어야 제맛", "한강 산책", "방에서 혼자"),
    ]


def test_parse_drops_too_long_text():
    long_question = "가" * 51
    assert parse(f"{long_question} | A | B") == []
    assert parse("질문 | " + "가" * 21 + " | B") == []


async def test_prompt_includes_stats_and_genre():
    llm = ScriptedLLM(RESPONSE)
    stats = GenreStats(12, [("잔나비", 5)], [("그리운", 4)], [("성수동", 3)], [("비", 2)])
    questions = await QuestionGenerator(llm).generate("인디음악", stats, count=1)
    assert questions == [("비 오는 날 인디는?", "잔나비", "검정치마")]
    assert "'인디음악' 장르" in llm.last_prompt
    assert "잔나비(5)" in llm.last_prompt and "성수동(3)" in llm.last_prompt


async def test_prompt_without_stats_asks_for_general_questions():
    llm = ScriptedLLM(RESPONSE)
    await QuestionGenerator(llm).generate("공통", GenreStats.empty(), count=2)
    assert "모든 사용자 공통" in llm.last_prompt
    assert "아직 저장된 기록이 없다" in llm.last_prompt


def _row(track, artist, place=None, weather=None):
    return SimpleNamespace(
        external_track_id=track, artist_name=artist, place_name=place, weather_condition=weather
    )


def test_aggregate_counts_only_the_genre_for_genre_groups():
    rows = [
        _row("t1", "잔나비", "성수동", "비"),
        _row("t1", "잔나비", "한강", "맑음"),
        _row("t2", "아이유", "성수동", "비"),
        _row("t3", "모르는곡"),
    ]
    moods = {
        "t1": SimpleNamespace(genre="인디음악", moods=["그리운"]),
        "t2": SimpleNamespace(genre="발라드", moods=["잔잔한"]),
    }
    indie = aggregate(rows, moods, "인디음악")
    assert indie.record_count == 2
    assert indie.top_artists == [("잔나비", 2)]
    assert indie.top_moods == [("그리운", 2)]

    common = aggregate(rows, moods, "공통")
    assert common.record_count == 4
    assert common.top_places == [("성수동", 2), ("한강", 1)]
