import math
from datetime import date, datetime

import pytest

from app.components.condition_scoring import (
    ConditionWeights,
    dropped_place_words,
    kept_place_words,
    resolve_conditions,
    score_rows,
    top_candidates,
)
from app.components.place_dictionary import PlaceEntry
from app.components.query_conditions import Condition
from tests.memory_fakes import make_row

WEIGHTS = ConditionWeights(location=2.0, other=1.0)
SEOUL_FOREST = PlaceEntry("서울숲", (), "park", 37.5433, 127.0332, 1.5)
HONGDAE = PlaceEntry("홍대", (), "downtown", 37.5575, 126.9248, 1.0)


def _score(row, *conditions, moods=None, music_similarity=None, min_similarity=0.4):
    scores = score_rows(
        [row],
        list(conditions),
        moods=moods or {},
        music_similarity=music_similarity or {},
        min_similarity=min_similarity,
        weights=WEIGHTS,
    )
    return scores[row.record_id]


# ---- 조건 하나씩 -------------------------------------------------------------


def test_place_scores_by_distance_to_the_dictionary_center():
    cond = Condition("place", "서울숲 근처", places=(SEOUL_FOREST,))
    near = make_row(1, latitude=37.5440, longitude=127.0370)  # 약 0.34km
    far = make_row(2, latitude=37.5575, longitude=126.9248)  # 홍대, 약 9.7km

    assert 0.7 < _score(near, cond).score < 0.8
    assert _score(near, cond).matched == ("서울숲 근처",)
    assert _score(far, cond).score == 0.0
    assert _score(far, cond).missed == ("서울숲 근처",)


def test_place_without_coordinates_does_not_match():
    cond = Condition("place", "서울숲 근처", places=(SEOUL_FOREST,))

    assert _score(make_row(1, latitude=None, longitude=None), cond).score == 0.0


def test_several_places_use_the_nearest_one():
    cond = Condition("place", "서울숲·홍대 근처", places=(SEOUL_FOREST, HONGDAE))
    at_hongdae = make_row(1, latitude=37.5575, longitude=126.9248)

    assert _score(at_hongdae, cond).score == 1.0


def test_region_matches_by_code_prefix():
    cond = Condition("region", "성동구", prefixes=("11200",))

    assert _score(make_row(1, dong_code="1120011500"), cond).all_matched
    assert not _score(make_row(2, dong_code="1111010100"), cond).matched
    assert not _score(make_row(3, dong_code=None), cond).matched


def test_city_prefix_covers_all_its_districts():
    cond = Condition("region", "수원시", prefixes=("4111",))

    assert _score(make_row(1, dong_code="4111710100"), cond).all_matched  # 수원시 영통구
    assert _score(make_row(2, dong_code="4111110100"), cond).all_matched  # 수원시 장안구


def test_dong_matches_by_name_prefix_so_1ga_and_2ga_both_count():
    cond = Condition("dong", "성수동", names=("성수동",))

    assert _score(make_row(1, dong_name="성수동1가"), cond).all_matched
    assert _score(make_row(2, dong_name="성수동2가"), cond).all_matched
    assert not _score(make_row(3, dong_name="연남동"), cond).matched


def test_period_includes_both_end_dates():
    cond = Condition("period", "지난달", ranges=((date(2026, 9, 1), date(2026, 9, 30)),))

    assert _score(make_row(1, created_at=datetime(2026, 9, 1, 0, 0)), cond).all_matched
    assert _score(make_row(2, created_at=datetime(2026, 9, 30, 23, 59)), cond).all_matched
    assert not _score(make_row(3, created_at=datetime(2026, 10, 1, 0, 0)), cond).matched
    assert not _score(make_row(4, created_at=datetime(2026, 8, 31, 23, 59)), cond).matched


def test_season_uses_the_month_regardless_of_year():
    winter = Condition("season", "겨울", months=frozenset({11, 12, 1, 2}))

    assert _score(make_row(1, created_at=datetime(2025, 1, 15)), winter).all_matched
    assert _score(make_row(2, created_at=datetime(2024, 11, 3)), winter).all_matched
    assert not _score(make_row(3, created_at=datetime(2026, 10, 3)), winter).matched


def test_weather_and_missing_weather():
    rain = Condition("weather", "비 오는 날", weather=frozenset({"RAIN", "SHOWER"}))

    assert _score(make_row(1, weather_condition="RAIN"), rain).all_matched
    assert not _score(make_row(2, weather_condition="CLEAR"), rain).matched
    assert not _score(make_row(3, weather_condition=None), rain).matched


def test_weather_value_is_compared_ignoring_case_and_surrounding_spaces():
    cloudy = Condition("weather", "흐린 날", weather=frozenset({"CLOUDY", "OVERCAST"}))

    for value in ("CLOUDY", "cloudy", " Cloudy ", "overcast", "OVERCAST\n"):
        assert _score(make_row(1, weather_condition=value), cloudy).all_matched, value
    assert not _score(make_row(2, weather_condition="CLOUDS"), cloudy).matched
    assert not _score(make_row(3, weather_condition=""), cloudy).matched


def test_temperature_and_mood_ranges_include_bounds_and_skip_missing_values():
    cold = Condition("temperature", "추운 날", low=-50, high=5)
    sad = Condition("mood", "기분이 안 좋았던 때", low=-50, high=-20)

    assert _score(make_row(1, temperature=5.0), cold).all_matched
    assert not _score(make_row(2, temperature=5.1), cold).matched
    assert not _score(make_row(3, temperature=None), cold).matched
    assert _score(make_row(4, mood_score=-20), sad).all_matched
    assert not _score(make_row(5, mood_score=-19), sad).matched
    assert not _score(make_row(6, mood_score=None), sad).matched


# ---- 곡 무드 태그 ------------------------------------------------------------


def test_song_mood_uses_tags_when_the_track_has_them():
    cond = Condition("song_mood", "잔잔한 곡", names=("잔잔한",))
    row = make_row(1)

    assert _score(row, cond, moods={1: ("잔잔한", "차분한")}).all_matched
    # 태그가 있는데 다른 태그면, 곡 분위기 벡터가 비슷해도 맞다고 하지 않는다
    other = _score(row, cond, moods={1: ("신나는",)}, music_similarity={1: 0.9})
    assert not other.matched


def test_song_mood_falls_back_to_music_similarity_when_track_has_no_tags():
    cond = Condition("song_mood", "잔잔한 곡", names=("잔잔한",))
    row = make_row(1)

    assert _score(row, cond, moods={1: ()}, music_similarity={1: 0.5}).all_matched
    assert _score(row, cond, music_similarity={1: 0.4}).all_matched  # 경계 포함
    assert not _score(row, cond, music_similarity={1: 0.39}).matched
    assert not _score(row, cond).matched  # 유사도를 모르면 맞다고 하지 않는다


# ---- 여러 조건 ---------------------------------------------------------------


def test_score_is_weighted_average_with_location_counting_double():
    place = Condition("place", "서울숲 근처", places=(SEOUL_FOREST,))
    season = Condition("season", "가을", months=frozenset({9, 10}))
    # 서울숲 중심에 있고(장소 1.0 × 2), 가을이 아니다(0 × 1) → 2/3
    row = make_row(1, latitude=37.5433, longitude=127.0332, created_at=datetime(2026, 5, 3))

    result = _score(row, place, season)

    assert math.isclose(result.score, 2 / 3)
    assert result.matched == ("서울숲 근처",)
    assert result.missed == ("가을",)
    assert not result.all_matched


def test_all_matched_only_when_every_condition_matches():
    place = Condition("place", "서울숲 근처", places=(SEOUL_FOREST,))
    season = Condition("season", "가을", months=frozenset({9, 10}))
    row = make_row(1, latitude=37.5440, longitude=127.0370, created_at=datetime(2026, 10, 3))

    assert _score(row, place, season).all_matched


def test_no_conditions_gives_no_scores():
    assert (
        score_rows(
            [make_row(1)], [], moods={}, music_similarity={}, min_similarity=0.4, weights=WEIGHTS
        )
        == {}
    )


# ---- 지역·동 후보 확인 -------------------------------------------------------


def test_dong_not_in_users_records_is_dropped():
    rows = [make_row(1, dong_name="연남동"), make_row(2, dong_name="성수동1가")]
    conditions = [Condition("dong", "운동", names=("운동",))]

    assert resolve_conditions(conditions, rows) == []


def test_dong_keeps_only_names_the_user_has():
    rows = [make_row(1, dong_name="연남동")]
    conditions = [Condition("dong", "연남동·운동", names=("연남동", "운동"))]

    (cond,) = resolve_conditions(conditions, rows)

    assert cond.names == ("연남동",)
    assert cond.label == "연남동"


def test_ambiguous_region_keeps_only_prefixes_the_user_has():
    rows = [make_row(1, dong_code="1114011500"), make_row(2, dong_code="1111010100")]
    middle = Condition("region", "중구", prefixes=("11140", "26110", "27110"))

    (cond,) = resolve_conditions([middle], rows)

    assert cond.prefixes == ("11140",)


def test_official_region_without_records_is_kept_but_abbreviation_is_dropped():
    rows = [make_row(1, dong_code="1114011500")]
    official = Condition("region", "성동구", prefixes=("11200",), reliable=True)
    abbreviation = Condition("region", "공주", prefixes=("44150",), reliable=False)

    assert resolve_conditions([official], rows) == [official]
    assert resolve_conditions([abbreviation], rows) == []


def test_other_conditions_pass_through_unchanged():
    season = Condition("season", "가을", months=frozenset({9, 10}))

    assert resolve_conditions([season], []) == [season]


# ---- 후보 고르기 -------------------------------------------------------------


def test_top_candidates_picks_best_scores_not_already_known():
    season = Condition("season", "가을", months=frozenset({9, 10}))
    place = Condition("place", "서울숲 근처", places=(SEOUL_FOREST,))
    rows = [
        make_row(
            1, created_at=datetime(2026, 10, 1), latitude=37.5433, longitude=127.0332
        ),  # 둘 다
        make_row(2, created_at=datetime(2026, 10, 2), latitude=37.0, longitude=127.0),  # 가을만
        make_row(3, created_at=datetime(2026, 3, 2), latitude=37.0, longitude=127.0),  # 아무것도
        make_row(4, created_at=datetime(2026, 10, 3), latitude=37.0, longitude=127.0),  # 가을만
    ]
    scores = score_rows(
        rows, [place, season], moods={}, music_similarity={}, min_similarity=0.4, weights=WEIGHTS
    )

    assert top_candidates(scores, exclude=set(), limit=10) == [
        1,
        4,
        2,
    ]  # 같은 점수면 최근 기록 먼저
    assert top_candidates(scores, exclude={1}, limit=10) == [4, 2]
    assert top_candidates(scores, exclude=set(), limit=1) == [1]


@pytest.mark.parametrize("limit", [0])
def test_top_candidates_with_zero_limit_is_empty(limit):
    season = Condition("season", "가을", months=frozenset({10}))
    scores = score_rows(
        [make_row(1)], [season], moods={}, music_similarity={}, min_similarity=0.4, weights=WEIGHTS
    )

    assert top_candidates(scores, exclude=set(), limit=limit) == []


# ---- 코멘트로 장소 찾기 ------------------------------------------------------

SEOUL_FOREST_PARK = PlaceEntry("서울숲", ("서울숲공원",), "park", 37.5433, 127.0332, 1.5)
EWHA = PlaceEntry("이화여자대학교", ("이대", "이화여대"), "campus", 37.5643, 126.9469, 1.0)
AT_SEOUL_FOREST = {"latitude": 37.5440, "longitude": 127.0370}
AT_HONGDAE = {"latitude": 37.5575, "longitude": 126.9248}


def _resolved_scores(rows, *conditions):
    """질문 조건을 이 사용자의 기록으로 확인한 뒤 점수를 낸다 (서비스와 같은 순서)."""
    resolved = resolve_conditions(list(conditions), rows)
    return score_rows(
        rows, resolved, moods={}, music_similarity={}, min_similarity=0.4, weights=WEIGHTS
    )


def test_place_with_nearby_records_is_compared_by_coordinates_only():
    rows = [
        make_row(1, **AT_SEOUL_FOREST, comment=None),
        make_row(2, **AT_HONGDAE, comment="서울숲에 다녀온 날"),  # 멀리서 저장하고 코멘트에만 적음
    ]
    cond = Condition("place", "서울숲 근처", places=(SEOUL_FOREST_PARK,))

    (resolved,) = resolve_conditions([cond], rows)
    scores = _resolved_scores(rows, cond)

    assert resolved.names == ()  # 코멘트로 찾지 않는다
    assert scores[1].score > 0
    assert scores[2].score == 0.0


def test_place_without_nearby_records_is_searched_in_comments_by_name_and_alias():
    rows = [
        make_row(1, **AT_HONGDAE, comment="서울숲 산책하다가 들은 곡"),
        make_row(2, **AT_HONGDAE, comment="서울숲공원 벤치에서"),
        make_row(3, **AT_HONGDAE, comment="그냥 걸었다"),
        make_row(4, **AT_HONGDAE, comment=None),
    ]
    cond = Condition("place", "서울숲 근처", places=(SEOUL_FOREST_PARK,))

    (resolved,) = resolve_conditions([cond], rows)
    scores = _resolved_scores(rows, cond)

    assert resolved.names == ("서울숲", "서울숲공원")
    assert resolved.label == "서울숲 근처"  # 안내 문장에 쓰는 이름은 그대로
    assert scores[1].score == scores[2].score == 1.0
    assert scores[3].score == scores[4].score == 0.0


def test_only_the_places_without_nearby_records_get_comment_names():
    rows = [make_row(1, **AT_SEOUL_FOREST)]
    cond = Condition("place", "서울숲·홍대 근처", places=(SEOUL_FOREST_PARK, HONGDAE))

    (resolved,) = resolve_conditions([cond], rows)

    assert resolved.names == ("홍대",)  # 서울숲은 근처에 기록이 있으니 좌표로만


def test_comment_mention_follows_the_same_before_and_after_rule_as_the_question():
    rows = [
        make_row(1, **AT_SEOUL_FOREST, comment="이대로 가자"),  # '이대로'는 이대가 아니다
        make_row(2, **AT_SEOUL_FOREST, comment="이대 앞에서 만났다"),
        make_row(3, **AT_SEOUL_FOREST, comment="스이대 같은 이름"),  # 앞에 한글이 붙어 있다
    ]
    cond = Condition("place", "이화여자대학교 근처", places=(EWHA,))

    scores = _resolved_scores(rows, cond)

    assert scores[1].score == 0.0
    assert scores[2].score == 1.0
    assert scores[3].score == 0.0


def test_comment_mention_ignores_spacing_and_case():
    rows = [make_row(1, **AT_HONGDAE, comment="서 울 숲 갔음")]
    cond = Condition("place", "서울숲 근처", places=(SEOUL_FOREST_PARK,))

    assert _resolved_scores(rows, cond)[1].score == 1.0


# ---- 사전에 없는 장소 말 -----------------------------------------------------


def _unknown(*words):
    return Condition("place_text", "·".join(words), names=tuple(words))


def test_unknown_place_is_kept_only_if_some_comment_mentions_it():
    rows = [make_row(1, comment="광교산 단풍 예뻤다"), make_row(2, comment="그냥 걸었다")]
    asked = [_unknown("광교산", "북한산")]

    resolved = resolve_conditions(asked, rows)

    assert [c.names for c in resolved] == [("광교산",)]
    assert resolved[0].label == "광교산"
    assert kept_place_words(resolved) == ["광교산"]
    assert dropped_place_words(asked, resolved) == ["북한산"]


def test_unknown_place_nobody_wrote_about_is_dropped_so_it_cannot_hide_other_results():
    rows = [make_row(1, comment="그냥 걸었다"), make_row(2, comment=None)]
    asked = [_unknown("북한산")]

    assert resolve_conditions(asked, rows) == []
    assert dropped_place_words(asked, []) == ["북한산"]


def test_unknown_place_matches_by_containing_text_not_only_whole_words():
    rows = [
        make_row(1, comment="광교산입구에서 출발"),  # 뒤에 글자가 이어져도 잡는다
        make_row(2, comment="광교 산 정상"),
        make_row(3, comment="북한산 갔다"),
        make_row(4, comment=None),
    ]

    scores = _resolved_scores(rows, _unknown("광교산"))

    assert scores[1].score == scores[2].score == 1.0
    assert scores[3].score == scores[4].score == 0.0


def test_unknown_place_counts_as_a_location_condition_in_the_weighting():
    row = make_row(1, comment="광교산 단풍", created_at=datetime(2026, 5, 2, 9, 0))
    season = Condition("season", "가을", months=frozenset({9, 10}))

    score = _score(row, _unknown("광교산"), season)

    # 장소류 가중치 2, 계절 1 → (2×1 + 1×0) ÷ 3
    assert math.isclose(score.score, 2 / 3)
    assert score.matched == ("광교산",)
    assert score.missed == ("가을",)
