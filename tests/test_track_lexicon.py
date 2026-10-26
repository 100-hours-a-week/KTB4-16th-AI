import pytest

from app.components.track_lexicon import (
    TrackLexicon,
    artist_keys,
    artist_name_keys,
    artist_names,
    compact,
    title_keys,
    title_variants,
)
from app.db.repositories.memory_record_repository import TrackName


def _lexicon(*tracks: tuple[str | None, str | None]) -> TrackLexicon:
    return TrackLexicon.build(TrackName(title, artist) for title, artist in tracks)


# ---- 이름 다듬기 -------------------------------------------------------------


def test_artist_column_with_several_artists_is_split_one_by_one():
    assert artist_names("RESCENE, 리센느") == ["RESCENE", "리센느"]
    assert artist_names("A/B & C") == ["A", "B", "C"]
    assert artist_names(None) == [] and artist_names("") == []


def test_title_variants_drop_brackets_and_dash_suffixes():
    assert title_variants("밤편지 (Lullaby)") == ["밤편지 (Lullaby)", "밤편지"]
    assert title_variants("Hype Boy - Special Edition") == [
        "Hype Boy - Special Edition",
        "Hype Boy",
    ]
    assert title_variants("소나기") == ["소나기"]
    assert title_variants(None) == [] and title_variants("  ") == []


def test_compact_ignores_spaces_case_and_symbols():
    assert compact("Hype Boy") == compact("hype-boy") == "hypeboy"
    assert artist_keys("RESCENE, 리센느") == {"rescene", "리센느"}
    assert title_keys("밤편지 (Lullaby)") == {"밤편지lullaby", "밤편지"}


# ---- 질문에서 찾기 -----------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "리센느 노래 들었던 거 뭐 있었지",
        "리센느가 부른 노래",
        "리센느의 노래",
        "리센느노래 들은 날",  # 붙여 쓴 경우
        "RESCENE 노래",
        "rescene 노래",
    ],
)
def test_artist_is_found_in_the_question(question):
    artists, _ = _lexicon(("소나기", "RESCENE, 리센느")).find_artists(question)

    assert artists and artists[0] in ("RESCENE", "리센느")


def test_found_text_is_blanked_so_other_rules_cannot_take_it_again():
    lexicon = _lexicon(("소나기", "리센느"))

    artists, rest = lexicon.find_artists("리센느 노래 들은 날")

    assert artists == ["리센느"]
    assert "리센느" not in rest and "노래 들은 날" in rest


def test_title_is_found_by_its_core_name_and_with_josa_after_it():
    lexicon = _lexicon(("밤편지 (Lullaby)", "아이유"))

    assert lexicon.find_titles("밤편지 들은 날")[0] == ["밤편지"]
    assert lexicon.find_titles("밤편지라는 노래 들은 날")[0] == ["밤편지"]
    assert lexicon.find_titles("밤편지가 듣고 싶던 날")[0] == ["밤편지"]


def test_english_title_ignores_case_and_spacing():
    lexicon = _lexicon(("Hype Boy", "NewJeans"))

    assert lexicon.find_titles("hype boy 들은 날")[0] == ["Hype Boy"]
    assert lexicon.find_titles("hypeboy 들은 날")[0] == ["Hype Boy"]


@pytest.mark.parametrize(
    "question",
    [
        "이대로 끝나는 날 들은 노래",  # "이대"는 이름이 아니다
        "아이유씨 이야기",  # 뒤가 조사가 아닌 낱말
        "노래 들은 날",  # 일반어
        "뉴진스 노래",  # 사전에 없는 표기 (영문으로 저장됨)
    ],
)
def test_words_that_are_not_names_are_not_found(question):
    lexicon = _lexicon(("Hype Boy", "NewJeans"), ("밤편지", "아이유"), ("이대", "누구"))

    assert lexicon.find_artists(question)[0] == []
    if "이대" in question:
        assert lexicon.find_titles(question)[0] == []


def test_longer_name_wins_and_the_shorter_one_inside_it_is_not_found_again():
    lexicon = _lexicon(("Love wins all", "IU"), ("Love", "누군가"))

    titles, rest = lexicon.find_titles("love wins all 들은 날")

    assert titles == ["Love wins all"]
    assert "love" not in rest.lower()


def test_names_shorter_than_two_characters_and_generic_words_stay_out_of_the_lexicon():
    lexicon = _lexicon(("비", "박"), ("노래", "음악"), (None, None), ("", " "))

    assert not lexicon
    assert lexicon.find_artists("박 음악 비 노래")[0] == []
    assert lexicon.find_titles("박 음악 비 노래")[0] == []


def test_matches_report_the_span_and_whether_the_name_was_said_explicitly():
    lexicon = _lexicon(("가을", "누군가"), ("밤편지", "아이유"))

    plain = lexicon.matches("밤편지 들은 날")
    quoted = lexicon.matches("'가을' 노래 들은 날")
    called = lexicon.matches("가을이라는 노래 들은 날")

    assert [(m.kind, m.name, m.explicit) for m in plain] == [("title", "밤편지", False)]
    (q,) = quoted
    assert (q.name, q.explicit) == ("가을", True)
    assert "'가을'" == "'가을' 노래 들은 날"[q.start : q.end]  # 따옴표까지 자리에 포함
    (c,) = called
    assert (c.name, c.explicit) == ("가을", True)
    assert "가을이라는" == "가을이라는 노래 들은 날"[c.start : c.end]  # "이라는"까지 포함


@pytest.mark.parametrize("question", ['"가을"', "「가을」", "『가을』", "‘가을’"])
def test_names_wrapped_in_any_kind_of_quotes_are_explicit(question):
    (match,) = _lexicon(("가을", "누군가")).matches(f"{question} 들은 날")

    assert match.explicit


def test_empty_lexicon_is_falsy_and_finds_nothing():
    assert not TrackLexicon.build([])
    assert TrackLexicon.build([]).find_artists("아이유 노래") == ([], "아이유 노래")


# ---- 가수 별칭 (영어↔한글) ---------------------------------------------------


def test_other_spelling_finds_the_artist_and_reports_the_stored_name():
    lexicon = TrackLexicon.build([TrackName("소나기", "RESCENE")], {"rescene": ["리센느"]})

    found, blanked = lexicon.find_artists("리센느 노래 들었던 날")

    assert found == ["RESCENE"]  # 안내 문장에는 저장된 이름을 쓴다
    assert "리센느" not in blanked
    assert lexicon.find_artists("RESCENE 노래")[0] == ["RESCENE"]  # 저장된 이름도 그대로 찾는다


def test_aliases_apply_to_each_artist_of_a_several_artist_column():
    lexicon = TrackLexicon.build(
        [TrackName("곡", "로꼬, 펀치")], {"로꼬": ["Loco"], "펀치": ["Punch"]}
    )

    assert lexicon.find_artists("loco 노래")[0] == ["로꼬"]
    assert lexicon.find_artists("punch 노래")[0] == ["펀치"]


def test_aliases_of_artists_the_user_never_saved_do_not_leak_in():
    lexicon = TrackLexicon.build([TrackName("소나기", "RESCENE")], {"adele": ["아델"]})

    assert lexicon.find_artists("아델 노래 들었던 날") == ([], "아델 노래 들었던 날")


def test_too_short_or_generic_aliases_are_ignored():
    lexicon = TrackLexicon.build([TrackName("곡", "RESCENE")], {"rescene": ["가", "노래"]})

    assert not lexicon.find_artists("가 노래")[0]


def test_alias_said_with_the_name_marker_is_explicit():
    lexicon = TrackLexicon.build([TrackName("소나기", "RESCENE")], {"rescene": ["리센느"]})

    (match,) = lexicon.matches("리센느라는 가수")

    assert (match.kind, match.name, match.explicit) == ("artist", "RESCENE", True)


def test_artist_name_keys_split_columns_and_skip_empty_ones():
    tracks = [
        TrackName("곡", "RESCENE, 리센느"),
        TrackName("곡", None),
        TrackName("곡", "Hype Boy"),
    ]

    assert artist_name_keys(tracks) == {"rescene", "리센느", "hypeboy"}


# ---- 오타 고쳐 찾기 -----------------------------------------------------------

_SAVED = TrackLexicon.build(
    [
        TrackName("Hype Boy", "NewJeans"),
        TrackName("밤편지 (Lullaby)", "아이유"),
        TrackName("주저하는 연인들을 위해", "잔나비"),
        TrackName("여행", "볼빨간사춘기"),
        TrackName("선물", "멜로망스"),
        TrackName("봄", "태연"),
    ],
    {"newjeans": ["뉴진스"]},
)


def _fixed(text: str, lexicon: TrackLexicon = _SAVED) -> list[tuple[str, str, str]]:
    found, _ = lexicon.find_fuzzy(text)
    return [(f.kind, f.name, f.typed) for f in found]


@pytest.mark.parametrize(
    "question, expected",
    [
        ("뉴진즈 노래", ("artist", "NewJeans", "뉴진즈")),  # 별칭(뉴진스)과 자모 1개 차이
        ("뉴진즈가 부른 노래", ("artist", "NewJeans", "뉴진즈")),  # 조사는 떼고 비교
        ("아이우 노래", ("artist", "아이유", "아이우")),  # 모음 하나
        ("잔나빈 노래", ("artist", "잔나비", "잔나빈")),  # 받침 하나
        ("멜로망쓰 노래", ("artist", "멜로망스", "멜로망쓰")),
        ("newjens 들은 날", ("artist", "NewJeans", "newjens")),  # 영어는 글자 단위
        (
            "주저하는 연인들을 위헤",  # 띄어 쓴 긴 곡명은 오타 2개까지
            ("title", "주저하는 연인들을 위해", "주저하는 연인들을 위헤"),
        ),
    ],
)
def test_typos_find_the_saved_name(question, expected):
    assert _fixed(question) == [expected]


def test_found_typo_is_blanked_out_of_the_question():
    _, rest = _SAVED.find_fuzzy("뉴진즈 노래 들은 날")

    assert "뉴진즈" not in rest and "들은 날" in rest


@pytest.mark.parametrize(
    "question",
    [
        "태영 노래",  # 두 글자 이름(태연)은 고치지 않는다
        "기분 좋았던 날 노래 찾아줘",
        "친구랑 놀러 간 날",
        "혼자 걷던 날 듣던 노래",
        "퇴근길에 들은 노래",
        "카페에서 들은 노래",
    ],
)
def test_ordinary_words_are_not_turned_into_names(question):
    assert _fixed(question) == []


def test_two_equally_close_names_are_left_alone():
    lexicon = _lexicon(("곡1", "아이유"), ("곡2", "아이요"))

    assert _fixed("아이우 노래", lexicon) == []  # 아이유·아이요 둘 다 1개 차이
