import textwrap

import pytest

from app.components.place_dictionary import PlaceDictionary, PlaceEntry


def _entry(name, *aliases, type="park", lat=37.5, lng=127.0, radius=1.5) -> PlaceEntry:
    return PlaceEntry(name, tuple(aliases), type, lat, lng, radius)


def _dictionary() -> PlaceDictionary:
    return PlaceDictionary(
        [
            _entry("홍대", "홍대입구", "홍대 앞", type="downtown"),
            _entry("이화여자대학교", "이대", "이화여대", type="campus"),
            _entry("중앙대학교", "중앙대", "중대", type="campus"),
            _entry("연세대학교", "연대", "연세대", type="campus"),
            _entry("서울숲", "서울숲공원"),
            _entry("샤로수길", "서울대입구", type="downtown"),
            _entry("서울대학교", "서울대", type="campus"),
        ]
    )


def _names(text: str) -> list[str]:
    return [e.name for e in _dictionary().find(text)[0]]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("가을에 서울숲에서 들었던 노래", ["서울숲"]),
        ("서울숲 근처에서 들은 노래", ["서울숲"]),
        ("홍대 갔을 때 들은 노래", ["홍대"]),
        ("홍대앞에서 들은 노래", ["홍대"]),  # 공백 없이 써도 된다
        ("홍대 앞에서 들은 노래", ["홍대"]),
        ("이대에서 들은 노래", ["이화여자대학교"]),
        ("홍대랑 서울숲에서 들은 노래", ["홍대", "서울숲"]),
    ],
)
def test_finds_places_followed_by_particle_or_boundary(text, expected):
    assert sorted(_names(text)) == sorted(expected)


@pytest.mark.parametrize(
    "text",
    [
        "이대로 가면 안 되는데 들은 노래",  # "이대로" 속 "이대"
        "중대한 일이 있던 날 들은 노래",  # "중대한" 속 "중대"
        "연대감을 느꼈던 날 들은 노래",  # "연대감" 속 "연대"
        "스이대에서 들은 노래",  # 앞에 한글이 붙어 있다
        "서울숲속에서 들은 노래",  # 이름이 더 긴 단어의 앞부분
    ],
)
def test_does_not_match_inside_longer_words(text):
    assert _names(text) == []


def test_longer_name_wins_over_its_prefix():
    # "서울대입구"가 "서울대"보다 먼저 잡혀서 서울대학교로 오인되지 않는다
    assert _names("서울대입구에서 들은 노래") == ["샤로수길"]
    assert _names("서울대에서 들은 노래") == ["서울대학교"]


def test_matched_text_is_blanked_out_so_other_rules_cannot_reuse_it():
    _, remaining = _dictionary().find("가을에 서울숲에서 들은 노래")

    assert "서울숲" not in remaining
    assert "가을에" in remaining and "에서 들은 노래" in remaining
    assert len(remaining) == len("가을에 서울숲에서 들은 노래")  # 길이는 그대로


def test_same_place_found_by_two_names_is_returned_once():
    assert _names("홍대입구에서 홍대 앞까지 걸었던 날") == ["홍대"]


def test_no_place_in_question():
    found, remaining = _dictionary().find("비 오던 날 들은 노래")
    assert found == []
    assert remaining == "비 오던 날 들은 노래"


# ---- 파일 읽기·검증 -------------------------------------------------------


def _write(tmp_path, *parts: str):
    path = tmp_path / "places.yaml"
    path.write_text("".join(textwrap.dedent(p) for p in parts), encoding="utf-8")
    return path


_HEADER = """
    type_radius_km:
      park: 1.5
      downtown: 1.0
"""


def test_loads_entry_with_type_default_radius_and_override(tmp_path):
    path = _write(
        tmp_path,
        _HEADER,
        """
        places:
          - name: 서울숲
            aliases: [서울숲공원]
            type: park
            center: [37.5433, 127.0332]
          - name: 홍대
            type: downtown
            center: [37.5575, 126.9248]
            radius_km: 0.6
        """,
    )

    entries = {e.name: e for e in PlaceDictionary.load(path).entries}

    assert entries["서울숲"].radius_km == 1.5  # 유형 기본값
    assert entries["서울숲"].aliases == ("서울숲공원",)
    assert (entries["서울숲"].lat, entries["서울숲"].lng) == (37.5433, 127.0332)
    assert entries["홍대"].radius_km == 0.6  # 항목에 적은 값이 우선


@pytest.mark.parametrize(
    ("place", "message"),
    [
        ("- {name: 가, type: 없는유형, center: [37.5, 127.0]}", "유형"),
        ("- {name: 가, type: park, center: [127.0, 37.5]}", "범위"),  # 위경도를 뒤바꿈
        ("- {name: 가, type: park, center: [37.5]}", "center"),
        ("- {type: park, center: [37.5, 127.0]}", "이름"),
        ("- {name: 가, type: park, center: [37.5, 127.0], radius_km: -1}", "반경"),
    ],
)
def test_invalid_entry_is_rejected(tmp_path, place, message):
    path = _write(tmp_path, _HEADER, "places:\n  " + place + "\n")

    with pytest.raises(ValueError, match=message):
        PlaceDictionary.load(path)


def test_duplicate_name_or_alias_is_rejected_even_with_different_spacing(tmp_path):
    path = _write(
        tmp_path,
        _HEADER,
        """
        places:
          - {name: 홍대, aliases: [홍대 앞], type: park, center: [37.5, 127.0]}
          - {name: 신촌, aliases: [홍대앞], type: park, center: [37.5, 127.0]}
        """,
    )

    with pytest.raises(ValueError, match="중복"):
        PlaceDictionary.load(path)


def test_empty_dictionary_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="하나도"):
        PlaceDictionary.load(_write(tmp_path, _HEADER, "places: []\n"))


# ---- 실제 places.yaml ------------------------------------------------------


def test_real_places_file_is_valid_and_finds_known_places():
    dictionary = PlaceDictionary.load()  # 검증(범위·중복·유형)은 읽는 중에 이뤄진다

    assert len(dictionary.entries) >= 50
    found, _ = dictionary.find("가을에 서울숲에서 들었던 노래 뭐지?")
    assert [e.name for e in found] == ["서울숲"]
    found, _ = dictionary.find("반포한강공원에서 들은 노래")
    assert [e.name for e in found] == ["반포한강공원"]
    found, _ = dictionary.find("홍대 갔던 날")
    assert [e.name for e in found] == ["홍대"]
