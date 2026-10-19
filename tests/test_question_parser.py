import textwrap
from datetime import date

import pytest

from app.components.question_parser import QuestionParser
from app.components.question_rules import QuestionRules

TODAY = date(2026, 10, 12)


@pytest.fixture(scope="module")
def parser() -> QuestionParser:
    return QuestionParser.load_default()  # 실제 places.yaml, regions.yaml, question_rules.yaml


def _kinds(parser, text, today=TODAY):
    return [c.kind for c in parser.parse(text, today)]


def _one(parser, text, kind):
    found = [c for c in parser.parse(text, TODAY) if c.kind == kind]
    assert len(found) == 1, (text, [c.kind for c in parser.parse(text, TODAY)])
    return found[0]


# ---- 한 종류씩 --------------------------------------------------------------


def test_place(parser):
    cond = _one(parser, "가을에 서울숲에서 들었던 노래 뭐지?", "place")

    assert [p.name for p in cond.places] == ["서울숲"]
    assert cond.label == "서울숲 근처"


def test_region(parser):
    cond = _one(parser, "성동구에서 들은 노래", "region")

    assert cond.prefixes == ("11200",)
    assert cond.label == "성동구"
    assert cond.reliable


def test_dong_candidate(parser):
    cond = _one(parser, "연남동에서 들은 노래", "dong")

    assert cond.names == ("연남동",)


def test_dong_with_ga_suffix(parser):
    assert _one(parser, "성수동1가에서 들은 노래", "dong").names == ("성수동1가",)
    assert _one(parser, "종로3가에서 들은 노래", "dong").names == ("종로3가",)


def test_season_uses_months_of_the_year(parser):
    assert _one(parser, "가을에 들은 노래", "season").months == frozenset({9, 10})
    assert _one(parser, "겨울에 들은 노래", "season").months == frozenset({11, 12, 1, 2})
    assert _one(parser, "한여름에 들은 노래", "season").months == frozenset({6, 7, 8})


def test_bare_month_without_year(parser):
    cond = _one(parser, "10월에 들은 노래", "season")

    assert cond.months == frozenset({10})
    assert cond.label == "10월"


def test_year_with_month_is_a_period_not_a_season(parser):
    kinds = _kinds(parser, "작년 10월에 들은 노래")

    assert kinds == ["period"]


def test_period(parser):
    cond = _one(parser, "지난달에 들은 노래", "period")

    assert cond.ranges == ((date(2026, 9, 1), date(2026, 9, 30)),)


def test_weather(parser):
    assert _one(parser, "비 오던 날 들은 노래", "weather").weather == {
        "RAIN",
        "RAIN_SNOW",
        "SHOWER",
    }
    assert _one(parser, "눈 오는 날 들은 노래", "weather").weather == {"SNOW", "RAIN_SNOW"}
    assert _one(parser, "맑은 날 들은 노래", "weather").weather == {"CLEAR"}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("흐린 날 들은 음악", {"CLOUDY", "OVERCAST"}),
        ("날씨가 흐렸을 때 들은 노래", {"CLOUDY", "OVERCAST"}),
        ("흐릴 때 들은 노래", {"CLOUDY", "OVERCAST"}),
        ("흐림 날씨에 들은 노래", {"CLOUDY", "OVERCAST"}),
        ("우중충한 날 들은 노래", {"CLOUDY", "OVERCAST"}),
        ("구름이 많은 날 들은 노래", {"CLOUDY", "OVERCAST"}),
        ("먹구름 낀 하늘 아래서 들은 노래", {"CLOUDY", "OVERCAST"}),
        ("비 왔던 날 들은 노래", {"RAIN", "RAIN_SNOW", "SHOWER"}),
        ("비 올 때 들은 노래", {"RAIN", "RAIN_SNOW", "SHOWER"}),
        ("눈 왔던 날 들은 노래", {"SNOW", "RAIN_SNOW"}),
        ("눈이 올 때 들은 노래", {"SNOW", "RAIN_SNOW"}),
        ("날씨가 맑았을 때 들은 노래", {"CLEAR"}),
        ("맑은 하늘 아래서 들은 노래", {"CLEAR"}),
    ],
)
def test_weather_also_understands_past_tense_and_other_wordings(parser, question, expected):
    assert _one(parser, question, "weather").weather == expected


@pytest.mark.parametrize(
    "question",
    ["흐름이 좋은 노래", "첫눈에 반한 노래", "준비하면서 들은 노래", "기억이 흐려진 날 들은 노래"],
)
def test_weather_words_do_not_catch_unrelated_words(parser, question):
    assert "weather" not in _kinds(parser, question)


def test_temperature(parser):
    cond = _one(parser, "추웠던 날 들은 노래", "temperature")
    assert (cond.low, cond.high) == (-50, 5)
    cond = _one(parser, "더웠던 날 들은 노래", "temperature")
    assert (cond.low, cond.high) == (26, 60)


def test_mood_is_the_users_own_feeling(parser):
    cond = _one(parser, "우울했을 때 들은 노래", "mood")

    assert (cond.low, cond.high) == (-50, -20)
    assert "song_mood" not in _kinds(
        parser, "우울했을 때 들은 노래"
    )  # "우울"이 곡 무드로 또 잡히지 않는다


def test_song_mood_is_how_the_song_feels(parser):
    assert _one(parser, "잔잔한 노래 듣고 싶어", "song_mood").names == ("잔잔한",)
    assert _one(parser, "우울한 노래 듣던 날", "song_mood").names == ("우울한",)
    assert "mood" not in _kinds(parser, "우울한 노래 듣던 날")


def test_warm_means_temperature_only_with_a_weather_word(parser):
    assert _kinds(parser, "따뜻한 노래") == ["song_mood"]
    assert _kinds(parser, "따뜻한 날 들은 노래") == ["temperature"]


# ---- 여러 조건 ---------------------------------------------------------------


def test_many_conditions_in_one_question(parser):
    conditions = parser.parse("작년 가을 서울숲에서 우울했을 때 들은 잔잔한 노래", TODAY)

    assert [c.kind for c in conditions] == ["period", "place", "mood", "song_mood"]
    assert conditions[0].ranges == ((date(2025, 9, 1), date(2025, 10, 31)),)


def test_region_and_dong_together(parser):
    kinds = _kinds(parser, "성남시 신흥동에서 들은 노래")

    assert kinds == ["region", "dong"]


def test_each_text_is_used_once(parser):
    # "서울숲"은 장소로만, "서울"이 지역으로 또 잡히지 않는다
    assert _kinds(parser, "서울숲에서 들은 노래") == ["place"]
    # "서울 성동구"는 지역 하나
    assert _kinds(parser, "서울 성동구에서 들은 노래") == ["region"]


# ---- 오탐 방지 ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "그냥 노래 추천해줘",
        "준비가 안 된 날 들은 노래",  # "비가"가 아니다
        "첫눈에 반한 날 들은 노래",  # 눈이 오는 날이 아니다
        "비교해서 알려줘",
        "이대로 끝나는 날 들은 노래",
        "중대한 발표가 있던 날",
        "운동할 때 들은 노래",  # "운동"은 동 이름 모양이 아니다 (뒤에 조사가 없다)
        "감동적인 날 들은 노래",
    ],
)
def test_ordinary_sentences_give_no_conditions(parser, text):
    assert parser.parse(text, TODAY) == []


def test_non_dong_words_ending_in_dong_are_candidates_only(parser):
    # 파서는 "운동을"을 동 후보로 올린다. 이 사용자의 기록에 그런 동이 없으면 뒤 단계가 버린다.
    assert _one(parser, "운동을 하던 날 들은 노래", "dong").names == ("운동",)


def test_present_tense_is_the_songs_feeling_and_past_tense_is_the_users_feeling(parser):
    assert _kinds(parser, "신나는 노래") == ["song_mood"]
    assert _kinds(parser, "신났던 날 들은 노래") == ["mood"]


# ---- 규칙 파일 검증 ----------------------------------------------------------


def _rules(tmp_path, body: str):
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


_VALID = """
    seasons:
      봄: {months: [3, 4, 5], words: [봄]}
    weather:
      - {label: 비, conditions: [RAIN], words: ["비 오"]}
    temperature:
      - {label: 추움, range: [-50, 5], words: ["추웠"]}
    mood:
      - {label: 안 좋음, range: [-50, -20], words: ["우울했"]}
    song_moods:
      잔잔한: [잔잔]
    """


def test_valid_rules_load(tmp_path):
    rules = QuestionRules.load(_rules(tmp_path, _VALID))

    assert rules.season_months == {"봄": frozenset({3, 4, 5})}


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("[3, 4, 5]", "[3, 4, 13]", "months"),
        ("[RAIN]", "[HAIL]", "conditions"),
        ("[-50, 5]", "[5, -50]", "range"),
        ("잔잔한: [잔잔]", "없는태그: [잔잔]", "곡 무드 태그"),
        ('words: ["비 오"]', "words: []", "words"),
    ],
)
def test_invalid_rules_are_rejected(tmp_path, old, new, message):
    body = textwrap.dedent(_VALID).replace(old, new)
    path = tmp_path / "rules.yaml"
    path.write_text(body, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        QuestionRules.load(path)


def test_real_rules_file_only_uses_known_song_mood_tags_and_weather_values():
    rules = QuestionRules.load()  # 읽는 동안 검증된다

    assert len(rules.song_moods) == 12
    assert {r.name for r in rules.seasons} == {"봄", "여름", "가을", "겨울"}


# ---- 사전에 없는 장소 -------------------------------------------------------


def test_unknown_place_word_becomes_a_place_text_condition(parser):
    cond = _one(parser, "가을에 광교산에서 들은 노래", "place_text")

    assert cond.names == ("광교산",)
    assert cond.label == "광교산"
    assert _kinds(parser, "가을에 광교산에서 들은 노래") == ["season", "place_text"]


def test_dictionary_place_is_not_also_taken_as_an_unknown_place(parser):
    # 서울숲은 '숲'으로 끝나지만 사전에서 먼저 잡혀 지워진다
    assert _kinds(parser, "서울숲에서 들은 노래") == ["place"]


def test_known_and_unknown_places_can_be_asked_together(parser):
    assert _kinds(parser, "서울숲이랑 광교산에서 들은 노래") == ["place", "place_text"]


def test_ordinary_place_words_do_not_become_conditions(parser):
    assert _kinds(parser, "공원에서 들은 노래") == []
    assert _kinds(parser, "이 지역에서 들은 노래") == []
