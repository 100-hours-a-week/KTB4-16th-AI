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

    assert [m.record_id for m in res.matched_records] == [4, 1, 2, 3]
    assert [m.similarity_score for m in res.matched_records] == [0.75, 0.7, 0.4, 0.35]
    assert res.retry_count == 0
    assert res.answer_text.startswith("비슷한 기록 4개를 찾았어요.")


def test_default_limit_is_five():
    # 프론트 계약은 "records 최대 5개"이고 limit을 따로 받지 않는다
    assert _request().limit == 5


async def test_default_limit_cuts_results_to_five():
    comment = [hit(i, 0.9 - i * 0.01) for i in range(1, 9)]
    service = _service(FakeVectorSearch(comment=comment), _records(*range(1, 9)))

    res = await service.search(_request())

    assert [m.record_id for m in res.matched_records] == [1, 2, 3, 4, 5]


async def test_record_ids_are_numbers_not_strings():
    service = _service(FakeVectorSearch(**SEARCH_HITS), _records(1, 2, 3, 4))

    res = await service.search(_request())

    assert all(isinstance(m.record_id, int) for m in res.matched_records)


async def test_limit_from_request_is_respected():
    service = _service(FakeVectorSearch(**SEARCH_HITS), _records(1, 2, 3, 4))

    res = await service.search(_request(limit=1))

    assert [m.record_id for m in res.matched_records] == [4]


async def test_record_that_cannot_be_read_is_skipped_and_next_one_fills_in():
    """AI DB 벡터는 남아 있는데 백엔드에서 삭제된 자물쇠 — 4번을 건너뛰고 3개를 채운다."""
    service = _service(FakeVectorSearch(**SEARCH_HITS), _records(1, 2, 3))

    res = await service.search(_request(limit=3))

    assert [m.record_id for m in res.matched_records] == [1, 2, 3]


async def test_other_users_records_never_come_back():
    """벡터 검색과 상세 조회 모두 요청한 사용자 ID로 부른다 — 다른 사용자의 자물쇠는 안 읽힌다."""
    search = FakeVectorSearch(**SEARCH_HITS)
    records = FakeMemoryRecords({999: {i: make_record(i) for i in (1, 2, 3, 4)}})
    service = _service(search, records)

    res = await service.search(_request())

    assert res.matched_records == []
    assert res.answer_text == NO_RESULT_TEXT
    assert {user for _, user, _ in search.calls} == {USER}
    assert [user for user, _ in records.calls] == [USER]


async def test_nothing_above_min_similarity_returns_no_record_message_without_reading_details():
    search = FakeVectorSearch(comment=[hit(1, 0.2)], music=[hit(2, 0.3)])
    records = _records(1, 2)
    service = _service(search, records)

    res = await service.search(_request())

    assert res.matched_records == []
    assert res.answer_text == NO_RESULT_TEXT
    assert records.calls == []  # 읽을 후보가 없으면 백엔드 DB를 부르지 않는다


async def test_only_a_few_extra_details_are_read_beyond_limit():
    search = FakeVectorSearch(comment=[hit(i, 0.9 - i * 0.01) for i in range(1, 11)])
    records = _records(*range(1, 11))
    service = _service(search, records)

    await service.search(_request(limit=2))

    assert len(records.calls[0][1]) == 6  # limit(2) x 3


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

    track = res.matched_records[0].track
    assert track.external_track_id == "2uwDoSLzTVVmWjsmbo8jhX"
    assert track.spotify_uri == "spotify:track:2uwDoSLzTVVmWjsmbo8jhX"


async def test_place_has_only_legal_dong_name_and_no_coordinates():
    search = FakeVectorSearch(comment=[hit(1, 0.9)])
    service = _service(search, _records(1))

    res = await service.search(_request())

    place = res.matched_records[0].place
    assert (place.place_id, place.name, place.lat, place.lng) == (10, "오산동", None, None)
