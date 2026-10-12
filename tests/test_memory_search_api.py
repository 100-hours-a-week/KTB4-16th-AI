import pytest

from app.dependencies import get_memory_record_source, get_memory_vector_search
from tests.conftest import AUTH
from tests.memory_fakes import FakeMemoryRecords, FakeVectorSearch, hit, make_record

URL = "/api/memory-search"


def _wire(client, search: FakeVectorSearch, records: FakeMemoryRecords) -> None:
    client.app.dependency_overrides[get_memory_vector_search] = lambda: search
    client.app.dependency_overrides[get_memory_record_source] = lambda: records


def _four_records(user: int = 35) -> tuple[FakeVectorSearch, FakeMemoryRecords]:
    search = FakeVectorSearch(
        comment=[hit(1, 0.9), hit(2, 0.8), hit(3, 0.7), hit(4, 0.6)],
        music=[hit(1, 0.5), hit(4, 0.9)],
    )
    return search, FakeMemoryRecords({user: {i: make_record(i) for i in (1, 2, 3, 4)}})


def test_returns_matches_with_camel_case_fields_and_numeric_ids(client):
    _wire(client, *_four_records())

    # 조건(장소·날씨 등)이 없는 질문이라 1단계 기준선 그대로 동작한다
    res = client.post(URL, json={"userId": 35, "queryText": "그때 들은 노래"}, headers=AUTH)

    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"answerText", "matchedRecords", "retryCount"}
    # recordId는 문자열이 아니라 숫자(Long)로 나간다 — 프론트 계약과 공통 ID 규칙
    assert [m["recordId"] for m in body["matchedRecords"]] == [4, 1, 2, 3]
    first = body["matchedRecords"][0]
    assert first["track"]["title"] == "밤편지"
    assert first["track"]["spotifyUri"] == "spotify:track:2uwDoSLzTVVmWjsmbo8jhX"
    assert first["place"]["name"] == "오산동"
    assert first["similarityScore"] == 0.75
    assert first["createdAt"].startswith("2026-08-24T18:30")
    assert body["retryCount"] == 0


def test_limit_in_request_overrides_default(client):
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": 35, "queryText": "비", "limit": 4}, headers=AUTH)

    assert len(res.json()["matchedRecords"]) == 4


def test_no_match_is_200_with_empty_list_and_message(client):
    _wire(client, FakeVectorSearch(), FakeMemoryRecords({}))

    res = client.post(URL, json={"userId": 35, "queryText": "없는 기억"}, headers=AUTH)

    assert res.status_code == 200
    assert res.json()["matchedRecords"] == []
    assert "기록이 없어요" in res.json()["answerText"]


@pytest.mark.parametrize("user_id", [35, "35"])
def test_numeric_user_id_is_accepted_as_number_or_string(client, user_id):
    search, records = _four_records(user=35)
    _wire(client, search, records)

    res = client.post(URL, json={"userId": user_id, "queryText": "비"}, headers=AUTH)

    assert res.status_code == 200
    assert {user for _, user, _ in search.calls} == {35}


@pytest.mark.parametrize("user_id", ["u_1", 0, -3, ""])
def test_invalid_user_id_is_400(client, user_id):
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": user_id, "queryText": "비"}, headers=AUTH)

    assert res.status_code == 400
    assert res.json()["field"] == "userId"


@pytest.mark.parametrize("query", ["", "   "])
def test_blank_query_is_400(client, query):
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": 35, "queryText": query}, headers=AUTH)

    assert res.status_code == 400
    assert res.json()["field"] == "queryText"


@pytest.mark.parametrize("limit", [0, 21])
def test_limit_out_of_range_is_400(client, limit):
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": 35, "queryText": "비", "limit": limit}, headers=AUTH)

    assert res.status_code == 400
    assert res.json()["field"] == "limit"


def test_needs_token(client):
    res = client.post(URL, json={"userId": 35, "queryText": "비"})

    assert res.status_code == 401
