import pytest

from app.components.unknown_places import find_unknown_places, strip_particle

# ---- 조사 떼기 ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("광교산에서", "광교산"),
        ("아이유가", "아이유"),
        ("바다를", "바다"),
        ("카페는", "카페"),
        ("비도", "비도"),  # 떼면 1자가 남아서 그대로 둔다
        ("아이유", "아이유"),  # 조사처럼 보이는 글자로 끝나도 남는 게 2자 미만이면 안 뗀다
        ("에서", "에서"),
    ],
)
def test_particles_are_stripped_only_when_two_or_more_letters_remain(word, expected):
    assert strip_particle(word) == expected


# ---- 사전에 없는 장소 말 찾기 ------------------------------------------------


def test_word_ending_in_a_place_suffix_is_an_unknown_place():
    assert find_unknown_places("광교산에서 들은 노래 뭐지?") == ["광교산"]
    assert find_unknown_places("서울역에서 들은 노래") == ["서울역"]


def test_several_places_keep_their_order_and_duplicates_are_dropped():
    assert find_unknown_places("북한산이랑 관악산 그리고 북한산에서") == ["북한산", "관악산"]


@pytest.mark.parametrize(
    "text",
    [
        "공원에서 들은 노래",  # 끝말만으로 된 일반 낱말
        "도서관에서 공부하며 들은 노래",
        "이 지역에서 들은 노래",  # 끝말이 우연히 같은 낱말
        "길거리에서 들은 노래",
        "계산하다가 들은 노래",
        "산책하다가 들은 노래",  # 끝이 '산'이 아니다
        "그때 들은 노래",
    ],
)
def test_ordinary_words_are_not_places(text):
    assert find_unknown_places(text) == []


def test_text_blanked_by_other_conditions_leaves_nothing():
    # 질문 해석이 조건으로 잡은 글자를 공백으로 지운 뒤의 모습
    assert find_unknown_places("       에서        들은 노래") == []
