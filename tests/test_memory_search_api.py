from datetime import datetime

import pytest

from app.config import get_settings
from app.dependencies import get_memory_record_source, get_memory_vector_search
from tests.conftest import AUTH
from tests.memory_fakes import FakeMemoryRecords, FakeVectorSearch, hit, make_record, make_row

URL = "/api/memory-search"


@pytest.fixture(autouse=True)
def _fixed_min_similarity(monkeypatch):
    """이 파일의 가짜 유사도(0.5~0.9)는 기준값 0.40을 전제로 짠 값이다.
    기본 기준값은 임베딩 모델에 따라 바뀌므로(KURE 0.65) 여기서는 고정한다."""
    monkeypatch.setenv("MEMORY_SEARCH_MIN_SIMILARITY", "0.40")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _wire(client, search: FakeVectorSearch, records: FakeMemoryRecords) -> None:
    client.app.dependency_overrides[get_memory_vector_search] = lambda: search
    client.app.dependency_overrides[get_memory_record_source] = lambda: records


def _four_records(user: int = 35) -> tuple[FakeVectorSearch, FakeMemoryRecords]:
    search = FakeVectorSearch(
        comment=[hit(1, 0.9), hit(2, 0.8), hit(3, 0.7), hit(4, 0.6)],
        music=[hit(1, 0.5), hit(4, 0.9)],
    )
    return search, FakeMemoryRecords({user: {i: make_record(i) for i in (1, 2, 3, 4)}})


def test_returns_only_record_ids_in_score_order_and_a_sentence_without_a_count(client):
    _wire(client, *_four_records())

    # 조건(장소·날씨 등)이 없는 질문이라 벡터 검색 그대로 동작한다
    res = client.post(URL, json={"userId": 35, "queryText": "그때 들은 노래"}, headers=AUTH)

    assert res.status_code == 200
    body = res.json()
    assert set(body) == {"answer", "records", "suggestedRecords"}
    # recordId는 문자열이 아니라 숫자(Long)로, 점수 높은 순으로 나간다 — 카드 정보는 백엔드가 만든다
    assert body["records"] == [{"recordId": 4}, {"recordId": 1}, {"recordId": 2}, {"recordId": 3}]
    assert body["suggestedRecords"] == []  # 조건이 없는 질문에는 제안이 없다(필드는 항상 있음)
    assert body["answer"] == "비슷한 자물쇠를 찾았어요."  # 건수 숫자 없음 (건수는 records.length)


def test_limit_in_request_overrides_default(client):
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": 35, "queryText": "비", "limit": 4}, headers=AUTH)

    assert len(res.json()["records"]) == 4


def test_no_match_is_200_with_empty_list_and_message(client):
    _wire(client, FakeVectorSearch(), FakeMemoryRecords({}))

    res = client.post(URL, json={"userId": 35, "queryText": "없는 기억"}, headers=AUTH)

    assert res.status_code == 200
    assert res.json()["records"] == []
    assert "자물쇠가 없어요" in res.json()["answer"]


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


@pytest.mark.parametrize("limit", [0, 51])
def test_limit_out_of_range_is_400(client, limit):
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": 35, "queryText": "비", "limit": limit}, headers=AUTH)

    assert res.status_code == 400
    assert res.json()["field"] == "limit"


def test_needs_token(client):
    res = client.post(URL, json={"userId": 35, "queryText": "비"})

    assert res.status_code == 401


def test_record_has_exactly_one_field_record_id(client):
    """명세 3.2: records[]는 { recordId }뿐이다. 카드 필드나 점수가 새로 끼어들면 계약이 깨진다."""
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": 35, "queryText": "그때 들은 노래"}, headers=AUTH)

    assert all(set(record) == {"recordId"} for record in res.json()["records"])


@pytest.mark.parametrize("name", ["query", "queryText"])
def test_both_the_backend_name_query_and_the_old_name_query_text_are_accepted(client, name):
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": 35, name: "그때 들은 노래"}, headers=AUTH)

    assert res.status_code == 200
    assert len(res.json()["records"]) == 4


@pytest.mark.parametrize("name", ["query", "queryText"])
def test_blank_query_is_400_and_names_the_field_that_was_sent(client, name):
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": 35, name: "   "}, headers=AUTH)

    assert res.status_code == 400
    assert res.json()["field"] == name


def test_missing_query_is_400(client):
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": 35}, headers=AUTH)

    assert res.status_code == 400


def test_suggested_records_have_only_the_record_id_and_the_matched_conditions(client):
    # 가을×서울숲을 모두 맞는 기록은 없고, 봄×서울숲(1번)과 가을×홍대(2번)가 일부만 맞는다
    forest = {"latitude": 37.5440, "longitude": 127.0370}
    hongdae = {"latitude": 37.5575, "longitude": 126.9248}
    rows = [
        make_row(1, created_at=datetime(2026, 5, 2, 9, 0), **forest),
        make_row(2, created_at=datetime(2026, 10, 9, 9, 0), **hongdae),
    ]
    details = {i: make_record(i) for i in (1, 2)}
    _wire(client, FakeVectorSearch(), FakeMemoryRecords({35: details}, {35: rows}))

    res = client.post(
        URL, json={"userId": 35, "query": "가을에 서울숲에서 들은 노래"}, headers=AUTH
    )

    body = res.json()
    assert body["records"] == []
    assert body["answer"] == "서울숲 근처, 가을 조건에 모두 맞는 자물쇠는 없어요."  # 목록·건수 없음
    by_id = {s["recordId"]: s for s in body["suggestedRecords"]}
    assert by_id[1]["matchedConditions"] == ["서울숲 근처"]
    assert by_id[2]["matchedConditions"] == ["가을"]
    # 제안 항목도 ID와 맞은 조건 이름뿐이다 (명세 3.2)
    assert all(set(item) == {"recordId", "matchedConditions"} for item in body["suggestedRecords"])


def test_the_response_matches_the_spec_example_shape(client):
    """명세서 3.2의 응답 예시와 같은 모양: answer, records[{recordId}], suggestedRecords[...]."""
    forest = {"latitude": 37.5440, "longitude": 127.0370}
    rows = [
        make_row(1, created_at=datetime(2026, 10, 2, 9, 0), **forest),
        make_row(2, created_at=datetime(2026, 5, 2, 9, 0), **forest),
    ]
    details = {i: make_record(i) for i in (1, 2)}
    _wire(client, FakeVectorSearch(), FakeMemoryRecords({35: details}, {35: rows}))

    body = client.post(
        URL, json={"userId": 35, "query": "가을에 서울숲에서 들은 노래"}, headers=AUTH
    ).json()

    assert body == {
        "answer": "서울숲 근처, 가을 조건에 맞는 자물쇠를 찾았어요.",
        "records": [{"recordId": 1}],
        "suggestedRecords": [{"recordId": 2, "matchedConditions": ["서울숲 근처"]}],
    }
