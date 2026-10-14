from datetime import datetime

import pytest

from app.dependencies import get_memory_record_source, get_memory_vector_search
from tests.conftest import AUTH
from tests.memory_fakes import FakeMemoryRecords, FakeVectorSearch, hit, make_record, make_row

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
    # 백엔드 명세(answer + records)와 겹치는 이름은 백엔드 이름을 쓴다
    assert set(body) == {"answer", "records", "suggestedRecords"}
    # recordId는 문자열이 아니라 숫자(Long)로 나간다 — 프론트 계약과 공통 ID 규칙
    assert [m["recordId"] for m in body["records"]] == [4, 1, 2, 3]
    assert body["suggestedRecords"] == []  # 조건이 없는 질문에는 제안이 없다(필드는 항상 있음)
    first = body["records"][0]
    assert first["musicTrackId"] == 3
    assert (first["title"], first["artistName"]) == ("밤편지", "아이유")
    assert (first["placeId"], first["placeName"]) == (10, "오산동")
    assert first["date"] == "2026.08.24"  # 한국시간 기준 "YYYY.MM.DD"
    assert first["weatherCondition"] == "RAIN"  # 카드의 날씨 아이콘용
    assert first["albumImageUrl"] == "https://img.example/cover.jpg"  # 카드 썸네일용
    assert first["externalTrackId"] == "2uwDoSLzTVVmWjsmbo8jhX"
    assert first["similarityScore"] == 0.75


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


def test_record_has_exactly_the_agreed_fields_flat_without_nesting(client):
    _wire(client, *_four_records())

    res = client.post(URL, json={"userId": 35, "queryText": "그때 들은 노래"}, headers=AUTH)

    record = res.json()["records"][0]
    assert set(record) == {
        # 백엔드 명세와 겹치는 필드 — 백엔드 이름 그대로
        "recordId",
        "musicTrackId",
        "title",
        "artistName",
        "placeId",
        "placeName",
        "date",
        # 백엔드 명세 records에 추가가 필요한 필드 — 카드의 날씨 아이콘과 앨범 썸네일용
        "weatherCondition",
        "albumImageUrl",
        # 백엔드 명세에 없는 AI 전용 필드 — 백엔드가 안 써도 된다
        "externalTrackId",
        "externalUrl",
        "similarityScore",
    }


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


def test_suggested_records_have_the_card_fields_and_the_matched_conditions(client):
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
    assert (
        body["answer"] == "서울숲 근처, 가을 조건에 모두 맞는 자물쇠는 없어요."
    )  # 목록도 제안 문구도 없다
    by_id = {s["recordId"]: s for s in body["suggestedRecords"]}
    assert by_id[1]["matchedConditions"] == ["서울숲 근처"]
    assert by_id[2]["matchedConditions"] == ["가을"]
    # records와 같은 카드 필드 + matchedConditions
    assert "matchedConditions" in by_id[1]
    for card_field in (
        "title",
        "artistName",
        "placeName",
        "date",
        "weatherCondition",
        "albumImageUrl",
    ):
        assert card_field in by_id[1]
