import textwrap

import pytest

from app.components.region_table import RegionTable


@pytest.fixture(scope="module")
def table() -> RegionTable:
    return RegionTable.load()  # 실제 app/regions.yaml — 읽는 동안 형식 검증도 된다


def _prefixes(table: RegionTable, text: str) -> list[tuple[str, ...]]:
    return [hit.prefixes for hit in table.find(text)[0]]


def test_sigungu_name_gives_five_digit_prefix(table):
    hits, _ = table.find("성동구에서 들은 노래")

    assert len(hits) == 1
    assert hits[0].prefixes == ("11200",)
    assert hits[0].text == "성동구"
    assert hits[0].reliable


def test_sido_name_and_short_name_give_two_digit_prefix(table):
    assert _prefixes(table, "서울에서 들은 노래") == [("11",)]
    assert _prefixes(table, "서울특별시에서 들은 노래") == [("11",)]
    assert _prefixes(table, "경기도에서 들은 노래") == [("41",)]
    assert _prefixes(table, "경기에서 들은 노래") == [("41",)]


def test_city_with_districts_matches_all_its_districts_by_four_digit_prefix(table):
    assert _prefixes(table, "수원시에서 들은 노래") == [("4111",)]
    # 구까지 말하면 그 구만
    assert _prefixes(table, "수원시 영통구에서 들은 노래") == [("41117",)]
    assert _prefixes(table, "화성시 동탄구에서 들은 노래") == [("41597",)]


def test_district_code_sample_from_backend_matches_hwaseong(table):
    # 백엔드 예시 병점동 코드 4159012700은 공식표에서 옛 화성시 반송동 — 시 단위(4159)로는 맞는다
    (hit,), _ = table.find("화성시에서 들은 노래")
    assert "4159012700".startswith(hit.prefixes[0])


def test_same_name_in_many_cities_returns_all_candidates(table):
    (hit,), _ = table.find("중구에서 들은 노래")

    assert "11140" in hit.prefixes  # 서울 중구
    assert len(hit.prefixes) >= 4  # 부산·대구·인천·대전·울산 등에도 있다


def test_sido_qualified_name_picks_exactly_one(table):
    assert _prefixes(table, "서울 중구에서 들은 노래") == [("11140",)]


def test_city_name_without_suffix_is_found_but_marked_unreliable(table):
    (hit,), _ = table.find("수원에서 들은 노래")

    assert hit.prefixes == ("4111",)
    assert not hit.reliable


@pytest.mark.parametrize(
    "text",
    [
        "서울숲에서 들은 노래",  # "서울숲"은 서울이 아니다
        "서울역 근처에서 들은 노래",
        "서울대입구에서 들은 노래",
        "중구난방으로 틀어놓은 노래",
        "공주님이 나오는 영화에서 들은 노래",  # "공주" 뒤에 한글이 이어진다
    ],
)
def test_does_not_match_region_names_inside_longer_words(table, text):
    assert table.find(text)[0] == []


def test_matched_text_is_blanked_out(table):
    _, remaining = table.find("성동구에서 들은 노래")

    assert "성동구" not in remaining
    assert len(remaining) == len("성동구에서 들은 노래")


def test_longer_qualified_name_is_consumed_before_the_shorter_one(table):
    hits, _ = table.find("서울 성동구에서 들은 노래")

    assert [h.prefixes for h in hits] == [("11200",)]  # "서울"이 따로 또 잡히지 않는다


def test_real_file_has_expected_shape(table):
    assert {"11", "41"} <= {s.code for s in table.sido}
    seoul = [g for g in table.sigungu if g.sido == "11"]
    assert len(seoul) == 25
    assert all(len(g.prefix) == 5 for g in seoul)


def _write(tmp_path, *parts: str):
    path = tmp_path / "regions.yaml"
    path.write_text("".join(textwrap.dedent(p) for p in parts), encoding="utf-8")
    return path


_SIDO = """
    sido:
      - {code: "11", name: 서울특별시, aliases: [서울]}
    """


@pytest.mark.parametrize(
    ("sigungu", "message"),
    [
        ('- {prefix: "12345", sido: "11", name: 가}', "시도 코드"),  # 서울(11)로 시작하지 않음
        ('- {prefix: "1111", sido: "99", name: 가}', "시도 코드"),  # 없는 시도
        ('- {prefix: "111", sido: "11", name: 가}', "올바르지"),  # 자리 수가 맞지 않음
        ('- {prefix: "11110", sido: "11"}', "올바르지"),  # 이름 없음
    ],
)
def test_invalid_region_file_is_rejected(tmp_path, sigungu, message):
    path = _write(tmp_path, _SIDO, "sigungu:\n  " + sigungu + "\n")

    with pytest.raises(ValueError, match=message):
        RegionTable.load(path)


def test_duplicate_prefix_is_rejected(tmp_path):
    path = _write(
        tmp_path,
        _SIDO,
        """
        sigungu:
          - {prefix: "11110", sido: "11", name: 종로구}
          - {prefix: "11110", sido: "11", name: 중구}
        """,
    )

    with pytest.raises(ValueError, match="중복"):
        RegionTable.load(path)
