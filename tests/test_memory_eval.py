"""기억 검색 정확도 측정 도구(scripts/memory_eval)의 계산 부분 시험 — 외부 파일·모델 없이 돈다."""

import math
from datetime import date

from app.components.place_dictionary import PlaceDictionary, PlaceEntry
from scripts.memory_eval.build_dataset import matches
from scripts.memory_eval.run_eval import InMemoryVectorSearch, random_hit_probability, wilson


def _record(**overrides) -> dict:
    record = {
        "record_id": 1,
        "created_at": "2026-09-10T18:30:00",
        "weather": "RAIN",
        "temperature": 18.0,
        "mood_score": -30,
        "dong_name": "성수동1가",
        "dong_code": "1120011500",
        "lat": 37.5440,
        "lng": 127.0370,
        "music_moods": ["잔잔한"],
    }
    record.update(overrides)
    return record


PLACES = PlaceDictionary([PlaceEntry("서울숲", (), "park", 37.5433, 127.0332, 1.5)])


def test_random_hit_probability():
    assert random_hit_probability(25, 0) == 0.0
    assert math.isclose(random_hit_probability(25, 1), 3 / 25)
    assert random_hit_probability(25, 25) == 1.0
    assert random_hit_probability(25, 10) > random_hit_probability(25, 5)


def test_wilson_interval_contains_the_estimate_and_stays_in_range():
    low, high = wilson(26, 27)

    assert 0.0 <= low < 26 / 27 < high <= 1.0
    assert wilson(0, 0) == (0.0, 0.0)


def test_all_filter_conditions_must_hold():
    rec = _record()

    assert matches(rec, {"dong_prefix": "성수동", "weather": ["RAIN", "SHOWER"]}, PLACES)
    assert not matches(rec, {"dong_prefix": "성수동", "weather": ["CLEAR"]}, PLACES)
    assert not matches(rec, {"dong_prefix": "연남동"}, PLACES)


def test_each_filter_kind():
    rec = _record()

    assert matches(rec, {"near": "서울숲"}, PLACES)
    assert matches(rec, {"region_prefix": "11"}, PLACES)
    assert not matches(rec, {"region_prefix": "41"}, PLACES)
    assert matches(rec, {"date_range": [date(2026, 9, 1), date(2026, 9, 30)]}, PLACES)
    assert not matches(rec, {"date_range": [date(2026, 10, 1), date(2026, 10, 31)]}, PLACES)
    assert matches(rec, {"months": [9, 10]}, PLACES) and not matches(rec, {"months": [3]}, PLACES)
    assert matches(rec, {"temp": [16, 25]}, PLACES) and not matches(rec, {"temp": [26, 60]}, PLACES)
    assert matches(rec, {"mood": [-50, -20]}, PLACES) and not matches(
        rec, {"mood": [20, 50]}, PLACES
    )
    assert matches(rec, {"song_mood": "잔잔한"}, PLACES)
    assert not matches(rec, {"song_mood": "신나는"}, PLACES)


async def test_in_memory_search_ranks_by_cosine_and_stays_inside_the_user():
    vectors = {"가": [1.0, 0.0], "나": [0.0, 1.0], "다": [0.8, 0.6], "질문": [1.0, 0.0]}
    users = {
        1: [
            {"record_id": 1, "external_track_id": "a", "comment": "가", "music_mood_text": "나"},
            {"record_id": 2, "external_track_id": "b", "comment": "다", "music_mood_text": "나"},
            {"record_id": 3, "external_track_id": "c", "comment": "나", "music_mood_text": "나"},
        ],
        2: [{"record_id": 9, "external_track_id": "z", "comment": "가", "music_mood_text": "가"}],
    }
    search = InMemoryVectorSearch(users, vectors)

    hits = await search.similar_by_comment(1, vectors["질문"], limit=2)

    assert [h.record_id for h in hits] == [
        1,
        2,
    ]  # 같은 방향 → 비슷한 방향 순서, 다른 사용자(9)는 안 나옴
    assert [round(h.similarity, 2) for h in hits] == [1.0, 0.8]
