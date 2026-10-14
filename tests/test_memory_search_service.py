from datetime import datetime

import pytest

from app.components.memory_answer import NO_RESULT_TEXT
from app.schemas.memory_search import MemorySearchRequest
from app.services.memory_search_service import MemorySearchService
from tests.memory_fakes import FakeMemoryRecords, FakeVectorSearch, StubEmbedder, hit, make_record

USER = 7


def _service(search, records, embedder=None) -> MemorySearchService:
    return MemorySearchService(
        embedder or StubEmbedder(),
        search,
        records,
        min_similarity=0.4,
        comment_weight=0.5,
        music_weight=0.5,
        candidates=20,
    )


def _request(**overrides) -> MemorySearchRequest:
    return MemorySearchRequest(**{"userId": USER, "queryText": "비 오던 날", **overrides})


def _records(*ids: int, user: int = USER) -> FakeMemoryRecords:
    return FakeMemoryRecords({user: {i: make_record(i) for i in ids}})


# 점수(가중 평균): 4번 0.75, 1번 0.70, 2번 0.40, 3번 0.35
SEARCH_HITS = {
    "comment": [hit(1, 0.9), hit(2, 0.8), hit(3, 0.7), hit(4, 0.6)],
    "music": [hit(1, 0.5), hit(4, 0.9)],
}


async def test_returns_all_matches_in_fused_order_when_fewer_than_default_limit():
    service = _service(FakeVectorSearch(**SEARCH_HITS), _records(1, 2, 3, 4))

    res = await service.search(_request())

    assert [m.record_id for m in res.records] == [4, 1, 2, 3]
    assert [m.similarity_score for m in res.records] == [0.75, 0.7, 0.4, 0.35]
    assert res.answer == "비슷한 자물쇠 4건을 찾았어요."  # 목록은 카드가 보여주므로 문장에 없다


def test_default_limit_is_fifty():
    # 프론트는 앞 5건을 먼저 보여주고 "더보기"에서 나머지를 보여주므로 전체(최대 50건)를 돌려준다
    assert _request().limit == 50


async def test_default_limit_returns_more_than_the_five_shown_first():
    comment = [hit(i, 0.9 - i * 0.01) for i in range(1, 9)]
    service = _service(FakeVectorSearch(comment=comment), _records(*range(1, 9)))

    res = await service.search(_request())

    assert [m.record_id for m in res.records] == [1, 2, 3, 4, 5, 6, 7, 8]


async def test_default_limit_cuts_results_to_fifty():
    comment = [hit(i, 0.95 - i * 0.005) for i in range(1, 61)]
    service = _service(FakeVectorSearch(comment=comment), _records(*range(1, 61)))

    res = await service.search(_request())

    assert [m.record_id for m in res.records] == list(range(1, 51))


async def test_record_ids_are_numbers_not_strings():
    service = _service(FakeVectorSearch(**SEARCH_HITS), _records(1, 2, 3, 4))

    res = await service.search(_request())

    assert all(isinstance(m.record_id, int) for m in res.records)


async def test_limit_from_request_is_respected():
    service = _service(FakeVectorSearch(**SEARCH_HITS), _records(1, 2, 3, 4))

    res = await service.search(_request(limit=1))

    assert [m.record_id for m in res.records] == [4]


async def test_record_that_cannot_be_read_is_skipped_and_next_one_fills_in():
    """AI DB 벡터는 남아 있는데 백엔드에서 삭제된 자물쇠 — 4번을 건너뛰고 3개를 채운다."""
    service = _service(FakeVectorSearch(**SEARCH_HITS), _records(1, 2, 3))

    res = await service.search(_request(limit=3))

    assert [m.record_id for m in res.records] == [1, 2, 3]


async def test_other_users_records_never_come_back():
    """벡터 검색과 상세 조회 모두 요청한 사용자 ID로 부른다 — 다른 사용자의 자물쇠는 안 읽힌다."""
    search = FakeVectorSearch(**SEARCH_HITS)
    records = FakeMemoryRecords({999: {i: make_record(i) for i in (1, 2, 3, 4)}})
    service = _service(search, records)

    res = await service.search(_request())

    assert res.records == []
    assert res.answer == NO_RESULT_TEXT
    assert {user for _, user, _ in search.calls} == {USER}
    assert [user for user, _ in records.calls] == [USER]


async def test_nothing_above_min_similarity_returns_no_record_message_without_reading_details():
    search = FakeVectorSearch(comment=[hit(1, 0.2)], music=[hit(2, 0.3)])
    records = _records(1, 2)
    service = _service(search, records)

    res = await service.search(_request())

    assert res.records == []
    assert res.answer == NO_RESULT_TEXT
    assert records.calls == []  # 읽을 후보가 없으면 백엔드 DB를 부르지 않는다


async def test_only_a_few_extra_details_are_read_beyond_limit():
    search = FakeVectorSearch(comment=[hit(i, 0.9 - i * 0.01) for i in range(1, 16)])
    records = _records(*range(1, 16))
    service = _service(search, records)

    await service.search(_request(limit=2))

    assert len(records.calls[0][1]) == 12  # limit(2) + 여유 10


async def test_query_is_embedded_once_and_each_search_uses_the_candidate_count():
    embedder = StubEmbedder()
    search = FakeVectorSearch(**SEARCH_HITS)
    service = _service(search, _records(1, 2, 3, 4), embedder)

    await service.search(_request(queryText="  비 오던 날  "))

    assert embedder.texts == ["비 오던 날"]
    assert sorted(search.calls) == [("comment", USER, 20), ("music", USER, 20)]


@pytest.mark.parametrize(
    "stored_id", ["2uwDoSLzTVVmWjsmbo8jhX", "spotify:track:2uwDoSLzTVVmWjsmbo8jhX"]
)
async def test_track_id_is_normalized_and_spotify_uri_is_built(stored_id):
    search = FakeVectorSearch(comment=[hit(1, 0.9)])
    records = FakeMemoryRecords({USER: {1: make_record(1, external_track_id=stored_id)}})
    service = _service(search, records)

    res = await service.search(_request())

    assert res.records[0].external_track_id == "2uwDoSLzTVVmWjsmbo8jhX"


async def test_place_has_only_the_legal_dong_name_and_no_coordinates():
    search = FakeVectorSearch(comment=[hit(1, 0.9)])
    service = _service(search, _records(1))

    res = await service.search(_request())

    item = res.records[0]
    assert (item.place_id, item.place_name) == (10, "오산동")
    assert not hasattr(item, "lat") and not hasattr(item, "lng")  # 위경도는 내보내지 않는다


async def test_date_is_year_month_day_with_dots_and_zero_padding_in_korean_time():
    search = FakeVectorSearch(comment=[hit(1, 0.9)])
    records = FakeMemoryRecords(
        {USER: {1: make_record(1, created_at=datetime(2026, 3, 4, 23, 50))}}
    )
    service = _service(search, records)

    res = await service.search(_request())

    assert res.records[0].date == "2026.03.04"  # DB 시각이 한국시간이라 변환 없이 날짜만


async def test_music_track_id_and_place_values_are_passed_through_including_missing_ones():
    search = FakeVectorSearch(comment=[hit(1, 0.9)])
    records = FakeMemoryRecords(
        {USER: {1: make_record(1, music_track_id=72, place_id=None, place_name=None)}}
    )
    service = _service(search, records)

    item = (await service.search(_request())).records[0]

    assert item.music_track_id == 72
    assert (item.place_id, item.place_name) == (None, None)  # 백엔드 places에 값이 없을 때


def test_request_accepts_the_backend_field_name_query():
    assert MemorySearchRequest(userId=7, query="  비 오던 날 ").query_text == "비 오던 날"


async def test_weather_condition_is_passed_through_including_a_missing_one():
    search = FakeVectorSearch(comment=[hit(1, 0.9), hit(2, 0.8)])
    records = FakeMemoryRecords(
        {
            USER: {
                1: make_record(1, weather_condition="SNOW"),
                2: make_record(2, weather_condition=None),
            }
        }
    )
    service = _service(search, records)

    items = (await service.search(_request())).records

    assert [i.weather_condition for i in items] == ["SNOW", None]


async def test_album_image_is_passed_for_the_card_thumbnail_and_empty_when_missing():
    search = FakeVectorSearch(comment=[hit(1, 0.9), hit(2, 0.8)])
    records = FakeMemoryRecords(
        {
            USER: {
                1: make_record(1, album_image_url="https://img/1.jpg"),
                2: make_record(2, album_image_url=""),
            }
        }
    )
    service = _service(search, records)

    items = (await service.search(_request())).records

    assert [i.album_image_url for i in items] == ["https://img/1.jpg", ""]
