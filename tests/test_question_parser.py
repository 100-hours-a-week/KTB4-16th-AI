import textwrap
from datetime import date

import pytest

from app.components.question_parser import QuestionParser
from app.components.question_rules import QuestionRules
from app.components.track_lexicon import TrackLexicon
from app.db.repositories.memory_record_repository import TrackName

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


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("10월 15일에 뭐 들었더라?", {(10, 15)}),
        ("10월15일에 들은 노래", {(10, 15)}),  # 붙여 써도
        ("2월 29일에 들은 노래", {(2, 29)}),  # 윤년의 날도 인정한다
        ("10월 15일과 11월 3일에 들은 노래", {(10, 15), (11, 3)}),
    ],
)
def test_month_and_day_without_a_year(parser, question, expected):
    cond = _one(parser, question, "day")

    assert cond.month_days == expected
    assert "season" not in _kinds(parser, question)  # "10월"이 달 조건으로 또 잡히지 않는다


def test_month_and_day_label_is_the_date_the_user_said(parser):
    assert _one(parser, "10월 15일에 뭐 들었더라?", "day").label == "10월 15일"


@pytest.mark.parametrize(
    ("question", "kind"),
    [
        ("10월에 들은 노래", "season"),  # 일 없이 달만
        ("2월 30일에 들은 노래", "season"),  # 없는 날짜는 달만 쓴다
        ("13월 5일에 들은 노래", None),  # 없는 달
    ],
)
def test_invalid_or_partial_dates_fall_back_to_the_month_rule(parser, question, kind):
    kinds = _kinds(parser, question)

    assert "day" not in kinds
    assert ("season" in kinds) == (kind == "season")


def test_year_word_with_month_and_day_is_that_single_day(parser):
    cond = _one(parser, "작년 10월 15일에 들은 노래", "period")  # 오늘 2026-10-12

    assert cond.ranges == ((date(2025, 10, 15), date(2025, 10, 15)),)
    assert "day" not in _kinds(parser, "작년 10월 15일에 들은 노래")


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


@pytest.mark.parametrize(
    ("question", "expected_hours"),
    [
        ("새벽에 들은 노래", {0, 1, 2, 3, 4}),
        ("아침에 들은 음악", {5, 6, 7, 8, 9, 10}),
        ("낮에 들은 노래", {11, 12, 13, 14, 15, 16}),
        ("오후에 들은 노래", {12, 13, 14, 15, 16, 17}),
        ("점심시간에 들은 노래", {11, 12, 13}),
        ("저녁에 들은 노래", {17, 18, 19, 20}),
        ("밤에 들은 노래", {21, 22, 23, 0, 1, 2, 3, 4}),  # 자정을 넘긴다 (새벽 포함)
        ("출근길에 들은 노래", {6, 7, 8, 9}),
        ("퇴근길에 들은 노래", {17, 18, 19, 20}),
        ("아침이나 저녁에 들은 노래", {5, 6, 7, 8, 9, 10, 17, 18, 19, 20}),  # 둘 중 하나
    ],
)
def test_time_of_day(parser, question, expected_hours):
    assert _one(parser, question, "time").hours == expected_hours


@pytest.mark.parametrize(
    ("question", "expected_days"),
    [
        ("주말에 들은 노래", {5, 6}),
        ("평일에 들은 노래", {0, 1, 2, 3, 4}),
        ("금요일에 들은 노래", {4}),
        ("토요일이나 일요일에 들은 노래", {5, 6}),
    ],
)
def test_weekday(parser, question, expected_days):
    assert _one(parser, question, "weekday").weekdays == expected_days


def test_time_and_weekday_combine_with_other_conditions(parser):
    kinds = _kinds(parser, "비 오는 날 금요일 저녁에 들은 노래")

    assert kinds == ["weather", "weekday", "time"]


def test_last_weekend_is_last_week_and_weekend(parser):
    conditions = parser.parse(
        "지난 주말에 들은 노래", TODAY
    )  # 2026-10-12(월) → 지난주는 10/5~10/11

    assert [c.kind for c in conditions] == ["period", "weekday"]
    assert conditions[0].ranges == ((date(2026, 10, 5), date(2026, 10, 11)),)
    assert conditions[0].label == "지난 주"  # "지난, 주말"처럼 앞 낱말만 나오지 않는다
    assert conditions[1].weekdays == {5, 6}


def test_this_weekend_is_this_week_and_weekend(parser):
    conditions = parser.parse("이번 주말에 들은 노래", TODAY)

    assert [c.kind for c in conditions] == ["period", "weekday"]
    assert conditions[0].ranges == ((date(2026, 10, 12), date(2026, 10, 18)),)


@pytest.mark.parametrize(
    "question",
    [
        "낮은 목소리의 노래",  # "낮"이 아니다
        "저녁노을 보며 들은 노래",
        "아침이슬 같은 노래",  # 곡 이름 같은 말
        "밤하늘을 보며 들은 노래",
        "한 주를 마무리하며 들은 노래",
    ],
)
def test_time_words_do_not_catch_unrelated_words(parser, question):
    kinds = _kinds(parser, question)

    assert "time" not in kinds and "weekday" not in kinds


@pytest.mark.parametrize(
    ("question", "expected_tags"),
    [
        ("발라드 들었던 날", ("발라드",)),
        ("힙합 노래 들은 날", ("랩/힙합",)),
        ("알앤비 들은 날", ("R&B/Soul",)),
        ("r&b 노래 들은 날", ("R&B/Soul",)),  # 대소문자 무시
        ("트로트 듣던 날", ("트로트",)),
        ("재즈 들은 날", ("재즈",)),
        ("팝송 들은 날", ("POP",)),
        ("일본 노래 들은 날", ("J-POP",)),
        ("록 음악 들은 날", ("록/메탈",)),
        ("영화 음악 들은 날", ("OST",)),
        ("발라드랑 재즈 들은 날", ("발라드", "재즈")),
    ],
)
def test_genre(parser, question, expected_tags):
    cond = _one(parser, question, "genre")

    assert cond.names == expected_tags
    assert cond.label.endswith(" 장르")


@pytest.mark.parametrize(
    "question",
    [
        "팝콘 먹으며 들은 노래",
        "기록을 보다가 들은 노래",
        "랩핑을 하며 들은 노래",
        "락커룸에서 들은 노래",
    ],
)
def test_genre_words_do_not_catch_unrelated_words(parser, question):
    assert "genre" not in _kinds(parser, question)


_NAMES = TrackLexicon.build(
    [
        TrackName("소나기", "RESCENE, 리센느"),
        TrackName("밤편지 (Lullaby)", "아이유"),
        TrackName("Hype Boy", "NewJeans"),
    ]
)


def test_artist_and_title_come_from_the_users_own_saved_tracks(parser):
    artist = parser.parse("리센느 노래 들었던 거 뭐 있었지", TODAY, _NAMES)
    title = parser.parse("밤편지 들은 날", TODAY, _NAMES)

    assert [(c.kind, c.label, c.names) for c in artist] == [("artist", "가수 리센느", ("리센느",))]
    assert [(c.kind, c.label, c.names) for c in title] == [("title", "곡 밤편지", ("밤편지",))]


def test_without_a_lexicon_names_are_not_looked_for(parser):
    assert parser.parse("리센느 노래 들었던 거 뭐 있었지", TODAY) == []
    assert parser.parse("리센느 노래", TODAY, TrackLexicon.build([])) == []


def test_names_combine_with_other_conditions_and_rules_take_text_first(parser):
    kinds = _kinds_with(parser, "비 오는 날 저녁에 아이유 노래 들은 거", _NAMES)

    assert kinds == ["weather", "time", "artist"]
    # 곡명 "소나기"는 날씨 규칙("소나기")이 먼저 글자를 가져간다
    assert _kinds_with(parser, "소나기 들은 날", _NAMES) == ["weather"]


def test_artist_name_is_not_taken_as_an_unknown_place(parser):
    # 가수 이름이 사전에 없는 장소 말로 또 잡히지 않는다
    lexicon = TrackLexicon.build([TrackName("노을", "광교산밴드")])

    assert _kinds_with(parser, "광교산밴드 노래", lexicon) == ["artist"]


_COLLISIONS = TrackLexicon.build(
    [
        TrackName("가을", "누군가"),
        TrackName("가을 아침", "아이유"),
        TrackName("소나기", "리센느"),
        TrackName("운동", "박"),
        TrackName("비 오는 날", "가수A"),
        TrackName("봄날", "방탄"),
    ]
)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # 애매하면 규칙(계절·날씨)이 이긴다
        ("가을에 들었던 노래", [("season", "가을")]),
        ("가을 노래 들은 거", [("season", "가을")]),
        ("소나기 들은 날", [("weather", "비 오는 날")]),
        ("비 오는 날 들은 노래", [("weather", "비 오는 날")]),
        ("봄날 들은 노래", [("season", "봄")]),
        # 분명히 이름이라고 말하면 이름이 이긴다
        ("가을이라는 노래 들은 날", [("title", "곡 가을")]),
        ("'가을' 노래 들은 날", [("title", "곡 가을")]),
        ("소나기라는 노래 들은 날", [("title", "곡 소나기")]),
        ('"비 오는 날"이라는 곡 들은 날', [("title", "곡 비 오는 날")]),
        ("봄날이라는 노래 들은 날", [("title", "곡 봄날")]),
        # 이름이 규칙 단어보다 의미 있게 길면 이름이 이긴다
        ("가을 아침 들은 날", [("title", "곡 가을 아침")]),
        # 단, 뒤에 "에"가 붙으면 "언제"를 말한 것이라 규칙이 이긴다
        ("가을 아침에 들은 노래", [("season", "가을"), ("time", "아침 시간")]),
        # 동 이름 후보("운동")는 나중에 걸러지므로 곡명이 먼저 가져간다
        ("운동을 하던 날 들은 노래", [("title", "곡 운동")]),
    ],
)
def test_name_and_rule_word_collisions_use_the_context_around_the_word(parser, question, expected):
    conditions = parser.parse(question, TODAY, _COLLISIONS)

    assert [(c.kind, c.label) for c in conditions] == expected


def test_artist_that_does_not_touch_any_rule_word_is_found_as_usual(parser):
    lexicon = TrackLexicon.build([TrackName("노을", "리센느")])

    assert _kinds_with(parser, "리센느 노래 들은 날", lexicon) == ["artist"]
    assert _kinds_with(parser, "비 오는 날 리센느 노래", lexicon) == ["weather", "artist"]


def _kinds_with(parser, text, lexicon):
    return [c.kind for c in parser.parse(text, TODAY, lexicon)]


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


_TIME_AND_WEEKDAY = """
    time_of_day:
      - {label: 밤, hours: [21, 5], words: ["밤에"]}
    weekday:
      - {label: 주말, days: [5, 6], words: ["주말"]}
    """


def test_time_and_weekday_sections_are_optional_and_load_when_present(tmp_path):
    without = QuestionRules.load(_rules(tmp_path, _VALID))
    with_them = QuestionRules.load(_rules(tmp_path, _VALID + _TIME_AND_WEEKDAY))

    assert without.times == () and without.weekdays == ()
    assert with_them.times[0].hours == {21, 22, 23, 0, 1, 2, 3, 4}
    assert with_them.weekdays[0].days == {5, 6}


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("[21, 5]", "[5, 5]", "hours"),  # 시작과 끝이 같다
        ("[21, 5]", "[24, 5]", "hours"),  # 시작은 0~23
        ("[21, 5]", "[21, 0]", "hours"),  # 끝은 1~24
        ("[5, 6]", "[5, 7]", "days"),  # 요일은 0~6
        ('words: ["밤에"]', "words: []", "words"),
    ],
)
def test_invalid_time_and_weekday_rules_are_rejected(tmp_path, old, new, message):
    body = (_VALID + _TIME_AND_WEEKDAY).replace(old, new)

    with pytest.raises(ValueError, match=message):
        QuestionRules.load(_rules(tmp_path, body))


_GENRE_SECTION = """
    genre:
      발라드: ["발라드"]
    """


def test_genre_section_is_optional_and_loads_when_present(tmp_path):
    without = QuestionRules.load(_rules(tmp_path, _VALID))
    with_it = QuestionRules.load(_rules(tmp_path, _VALID + _GENRE_SECTION))

    assert without.genres == ()
    assert [r.tag for r in with_it.genres] == ["발라드"]


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("발라드: [", "없는장르: [", "장르 목록"),  # GENRE_VOCAB에 없는 장르 이름
        ('["발라드"]', "[]", "words"),
    ],
)
def test_invalid_genre_rules_are_rejected(tmp_path, old, new, message):
    body = (_VALID + _GENRE_SECTION).replace(old, new)

    with pytest.raises(ValueError, match=message):
        QuestionRules.load(_rules(tmp_path, body))


def test_typo_of_a_saved_name_becomes_that_name_with_the_correction_kept(parser):
    conditions = parser.parse("아이우 노래 들은 날", TODAY, _NAMES)

    assert [(c.kind, c.label, c.names, c.corrections) for c in conditions] == [
        ("artist", "가수 아이유", ("아이유",), (("아이우", "아이유"),))
    ]


def test_typos_are_not_looked_for_when_an_exact_name_is_there(parser):
    conditions = parser.parse("아이유랑 하입보이 들은 날", TODAY, _NAMES)

    assert [(c.kind, c.names, c.corrections) for c in conditions] == [("artist", ("아이유",), ())]


def test_rules_take_their_words_before_typos_are_looked_for(parser):
    # "소나긔"를 곡명(소나기)으로 고치지 않는다 — 날씨 말이 아닌 남은 글자만 고쳐 본다
    kinds = _kinds_with(parser, "비 오는 날 아이우 노래", _NAMES)

    assert kinds == ["weather", "artist"]
