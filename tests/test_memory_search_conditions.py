"""조건이 있는 질문의 기억 검색 — 실제 규칙 파일(places/regions/question_rules)로 해석한다."""

from datetime import date, datetime

import pytest

from app.components.memory_answer import NO_RESULT_TEXT
from app.components.question_parser import QuestionParser
from app.schemas.memory_search import MemorySearchRequest
from app.services.memory_search_service import MemorySearchService
from tests.memory_fakes import (
    FakeMemoryRecords,
    FakeVectorSearch,
    StubEmbedder,
    hit,
    make_record,
    make_row,
)

USER = 7
TODAY = date(2026, 10, 12)


@pytest.fixture(scope="module")
def parser() -> QuestionParser:
    return QuestionParser.load_default()


def _service(parser, search, records) -> MemorySearchService:
    return MemorySearchService(
        StubEmbedder(),
        search,
        records,
        min_similarity=0.4,
        comment_weight=0.5,
        music_weight=0.5,
        candidates=20,
        parser=parser,
        condition_weight=0.5,
        today=lambda: TODAY,
    )


def _request(text: str, **overrides) -> MemorySearchRequest:
    return MemorySearchRequest(**{"userId": USER, "queryText": text, **overrides})


SEOUL_FOREST = {"latitude": 37.5440, "longitude": 127.0370}  # 서울숲 안쪽, 중심에서 약 0.34km
BANPO = {"latitude": 37.5089, "longitude": 126.9939}
HONGDAE = {"latitude": 37.5575, "longitude": 126.9248}


def _world():
    """서울숲×가을 1건, 서울숲만 1건, 가을만 1건, 둘 다 아닌 1건을 가진 사용자."""
    rows = [
        make_row(1, created_at=datetime(2026, 10, 3, 9, 0), dong_name="성수동1가", **SEOUL_FOREST),
        make_row(2, created_at=datetime(2026, 5, 2, 9, 0), dong_name="성수동2가", **SEOUL_FOREST),
        make_row(3, created_at=datetime(2025, 9, 14, 9, 0), dong_name="반포동", **BANPO),
        make_row(4, created_at=datetime(2026, 3, 5, 9, 0), dong_name="서교동", **HONGDAE),
    ]
    details = {
        1: make_record(1, created_at=datetime(2026, 10, 3, 9, 0), place_name="성수동1가"),
        2: make_record(2, created_at=datetime(2026, 5, 2, 9, 0), place_name="성수동2가"),
        3: make_record(3, created_at=datetime(2025, 9, 14, 9, 0), place_name="반포동"),
        4: make_record(4, created_at=datetime(2026, 3, 5, 9, 0), place_name="서교동"),
    }
    return rows, details


def _search(**kwargs) -> FakeVectorSearch:
    # 3·4번만 벡터 검색 상위 후보에 있고, 1·2번은 조건으로만 후보가 된다
    return FakeVectorSearch(
        comment=[hit(4, 0.95), hit(3, 0.6)],
        music=[hit(4, 0.9)],
        comment_similarity={1: 0.7},
        music_similarity={1: 0.8, 2: 0.5},
        **kwargs,
    )


async def test_exact_matches_are_results_and_partial_matches_are_suggested_in_text(parser):
    rows, details = _world()
    search = _search()
    records = FakeMemoryRecords({USER: details}, rows_by_user={USER: rows})
    service = _service(parser, search, records)

    res = await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    assert [m.record_id for m in res.matched_records] == [1]  # 가을 AND 서울숲
    lines = res.answer_text.splitlines()
    assert lines[0] == "조건에 모두 맞는 기록 1개를 찾았어요."
    assert "이런 기록을 찾으시는 건 아닌가요?" in lines
    assert "2. 2025년 9월 14일 · 반포동 · 아이유 - 밤편지 ('가을'만 맞아요)" in res.answer_text
    assert "1. 2026년 5월 2일 · 성수동2가 · 아이유 - 밤편지 ('서울숲 근처'만 맞아요)" in (
        res.answer_text
    )
    assert "2026년 3월" not in res.answer_text  # 조건을 하나도 못 맞춘 기록은 나오지 않는다
    assert res.retry_count == 0


async def test_records_found_only_by_conditions_get_their_similarity_looked_up(parser):
    rows, details = _world()
    search = _search()
    service = _service(parser, search, FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    assert search.similarity_calls == [(USER, [1, 2])]  # 이미 후보인 3·4번은 다시 구하지 않는다
    # 1번: 벡터 (0.7+0.8)/2=0.75, 조건 (장소 약 0.77×2 + 계절 1)/3 ≈ 0.85 → 최종 약 0.80
    assert 0.78 < res.matched_records[0].similarity_score < 0.82


async def test_suggestions_are_left_out_when_exact_matches_fill_the_limit(parser):
    rows, details = _world()
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래", limit=1))

    assert [m.record_id for m in res.matched_records] == [1]
    assert "이런 기록" not in res.answer_text


async def test_no_exact_match_returns_empty_list_and_suggestions_in_text(parser):
    rows, details = _world()
    rows = [r for r in rows if r.record_id != 1]  # 가을×서울숲 기록이 없는 사용자
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래"))

    assert res.matched_records == []
    assert res.answer_text.startswith("조건에 모두 맞는 기록은 없어요. 이런 기록을 찾으시는 건")


async def test_question_without_conditions_skips_the_condition_path(parser):
    rows, details = _world()
    records = FakeMemoryRecords({USER: details}, {USER: rows})
    search = _search()
    service = _service(parser, search, records)

    res = await service.search(_request("그때 들은 노래"))

    assert records.row_calls == []  # 자물쇠 정보를 읽지 않는다
    assert search.similarity_calls == [] and search.mood_calls == []
    assert res.answer_text.startswith("비슷한 기록")  # 1단계 기준선 문장


async def test_dong_word_the_user_never_visited_falls_back_to_plain_search(parser):
    rows, details = _world()
    records = FakeMemoryRecords({USER: details}, {USER: rows})
    service = _service(parser, _search(), records)

    res = await service.search(_request("운동을 하던 날 들은 노래"))

    assert records.row_calls == [(USER, 5000)]  # 후보를 확인하려고 읽었지만
    assert res.answer_text.startswith("비슷한 기록")  # 조건으로 남지 않아 기준선으로 처리
    assert [m.record_id for m in res.matched_records] == [4, 3]


async def test_dong_the_user_has_is_a_condition_and_matches_by_name_prefix(parser):
    rows, details = _world()
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("성수동에서 들은 노래"))

    assert sorted(m.record_id for m in res.matched_records) == [1, 2]  # 성수동1가·2가


async def test_region_without_any_records_gives_no_result_instead_of_unrelated_ones(parser):
    rows, details = _world()
    rows = [make_row(r.record_id, dong_code="1144010100") for r in rows]  # 모두 마포구
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("성동구에서 들은 노래"))

    assert res.matched_records == []
    assert res.answer_text == NO_RESULT_TEXT


async def test_region_matches_by_code_prefix(parser):
    rows, details = _world()
    rows = [
        make_row(1, dong_code="1120011500"),  # 성동구
        make_row(2, dong_code="1120011400"),  # 성동구
        make_row(3, dong_code="1144010100"),  # 마포구
        make_row(4, dong_code="1144010200"),  # 마포구
    ]
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("성동구에서 들은 노래"))

    assert sorted(m.record_id for m in res.matched_records) == [1, 2]


async def test_relative_period_uses_todays_date(parser):
    rows = [
        make_row(1, created_at=datetime(2026, 9, 10, 9, 0)),
        make_row(2, created_at=datetime(2026, 10, 1, 9, 0)),
        make_row(3, created_at=datetime(2026, 9, 30, 23, 0)),
    ]
    details = {i: make_record(i) for i in (1, 2, 3)}
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("지난달에 들은 노래"))  # 오늘 2026-10-12 → 9월

    assert sorted(m.record_id for m in res.matched_records) == [1, 3]


async def test_song_mood_uses_tags_and_falls_back_to_music_similarity_without_tags(parser):
    rows = [make_row(1), make_row(2), make_row(3), make_row(4)]
    details = {i: make_record(i) for i in (1, 2, 3, 4)}
    search = FakeVectorSearch(
        comment=[],
        music=[hit(2, 0.6), hit(4, 0.2)],  # 2번은 태그가 없지만 곡 분위기가 비슷하다
        moods={1: ("잔잔한",), 3: ("신나는",), 4: ()},
    )
    service = _service(parser, search, FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("잔잔한 노래 듣고 싶어"))

    assert search.mood_calls == [USER]
    # 1번: 태그 일치, 2번: 태그 없음+유사도 0.6 ≥ 0.4, 3번: 다른 태그, 4번: 태그 없음+유사도 0.2
    assert sorted(m.record_id for m in res.matched_records) == [1, 2]


async def test_mood_tags_are_not_read_when_the_question_has_no_song_mood(parser):
    rows, details = _world()
    search = _search()
    service = _service(parser, search, FakeMemoryRecords({USER: details}, {USER: rows}))

    await service.search(_request("가을에 서울숲에서 들었던 노래"))

    assert search.mood_calls == []


async def test_only_this_users_rows_are_read_and_other_users_get_nothing(parser):
    rows, details = _world()
    records = FakeMemoryRecords({USER: details}, rows_by_user={USER: rows})
    service = _service(parser, _search(), records)

    res = await service.search(_request("가을에 서울숲에서 들었던 노래", userId=999))

    assert records.row_calls == [(999, 5000)]
    assert res.matched_records == []
    assert res.answer_text == NO_RESULT_TEXT


async def test_exact_record_whose_details_cannot_be_read_is_skipped(parser):
    rows, details = _world()
    del details[1]  # AI DB에는 남아 있지만 백엔드에서 삭제된 자물쇠
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래"))

    assert res.matched_records == []
    assert res.answer_text.startswith("조건에 모두 맞는 기록은 없어요.")


async def test_combined_period_place_mood_question(parser):
    rows = [
        make_row(1, created_at=datetime(2025, 10, 3), mood_score=-30, **SEOUL_FOREST),
        make_row(2, created_at=datetime(2025, 10, 4), mood_score=30, **SEOUL_FOREST),  # 기분 좋음
        make_row(3, created_at=datetime(2026, 10, 3), mood_score=-30, **SEOUL_FOREST),  # 올해
    ]
    details = {i: make_record(i) for i in (1, 2, 3)}
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("작년 가을 서울숲에서 우울했을 때 들은 노래"))

    assert [m.record_id for m in res.matched_records] == [1]


# ---- 코멘트로 장소 찾기 / 사전에 없는 장소 -----------------------------------


def _comment_world():
    """가을·봄, 코멘트에 광교산이 있고 없는 기록을 섞어 둔 사용자."""
    rows = [
        make_row(1, created_at=datetime(2026, 10, 5, 9, 0), comment="광교산 단풍 예뻤다"),
        make_row(2, created_at=datetime(2026, 5, 2, 9, 0), comment="광교산 입구에서 출발"),
        make_row(3, created_at=datetime(2026, 9, 20, 9, 0), comment="카페에서 공부"),
        make_row(4, created_at=datetime(2026, 10, 9, 9, 0), comment=None),
    ]
    details = {r.record_id: make_record(r.record_id, created_at=r.created_at) for r in rows}
    return rows, details


async def test_unknown_place_is_searched_in_comments_and_combined_with_other_conditions(parser):
    rows, details = _comment_world()
    records = FakeMemoryRecords({USER: details}, {USER: rows})
    service = _service(parser, FakeVectorSearch(), records)

    res = await service.search(_request("가을에 광교산에서 들은 노래 뭐지?"))

    assert [m.record_id for m in res.matched_records] == [1]  # 가을 AND 코멘트에 광교산
    assert "'광교산'만 맞아요" in res.answer_text  # 2번(봄)은 장소만 맞아서 제안으로
    assert "'가을'만 맞아요" in res.answer_text  # 3·4번(가을)은 계절만 맞아서 제안으로
    assert res.answer_text.endswith(
        "'광교산'은 장소 목록에 없는 곳이라서, 기록에 적어 둔 코멘트에서 찾았어요."
    )


async def test_unknown_place_nobody_wrote_about_is_ignored_with_a_notice(parser):
    rows, details = _comment_world()
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 북한산에서 들은 노래"))

    # 위치 조건은 빠지고 가을 조건만으로 찾는다
    assert sorted(m.record_id for m in res.matched_records) == [1, 3, 4]
    assert res.answer_text.startswith("조건에 모두 맞는 기록 3개를 찾았어요.")
    assert res.answer_text.endswith(
        "'북한산'은 아직 모르는 장소라서 위치 조건은 적용하지 못했어요. "
        "동이나 구 이름(예: 성수동, 성동구)으로 물어보시면 더 정확하게 찾을 수 있어요."
    )


async def test_only_an_unknown_place_nobody_wrote_about_falls_back_to_plain_search_with_notice(
    parser,
):
    rows, details = _comment_world()
    records = FakeMemoryRecords({USER: details}, {USER: rows})
    search = FakeVectorSearch(comment=[hit(3, 0.9)])
    service = _service(parser, search, records)

    res = await service.search(_request("북한산에서 들은 노래"))

    assert [m.record_id for m in res.matched_records] == [3]  # 벡터 검색 결과 그대로
    assert res.answer_text.startswith("비슷한 기록 1개를 찾았어요.")
    assert "'북한산'은 아직 모르는 장소라서" in res.answer_text


async def test_notice_is_added_even_when_nothing_is_found(parser):
    rows, details = _comment_world()
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("북한산에서 들은 노래"))

    assert res.matched_records == []
    assert res.answer_text.startswith(NO_RESULT_TEXT)
    assert "'북한산'은 아직 모르는 장소라서" in res.answer_text


async def test_known_place_without_nearby_records_is_found_through_the_comment(parser):
    far = {"latitude": 37.5575, "longitude": 126.9248}  # 홍대 — 서울숲에서 멀다
    rows = [
        make_row(1, **far, comment="서울숲 산책하다 들은 곡"),
        make_row(2, **far, comment="합정에서 약속"),
    ]
    details = {i: make_record(i) for i in (1, 2)}
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("서울숲에서 들은 노래"))

    assert [m.record_id for m in res.matched_records] == [1]
    assert "장소 목록에 없는" not in res.answer_text  # 사전에 있는 장소라 안내할 것이 없다


async def test_known_place_with_nearby_records_ignores_comment_mentions_elsewhere(parser):
    rows = [
        make_row(1, **SEOUL_FOREST, comment=None),
        make_row(2, **HONGDAE, comment="서울숲에 다녀온 날"),
    ]
    details = {i: make_record(i) for i in (1, 2)}
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("서울숲에서 들은 노래"))

    assert [m.record_id for m in res.matched_records] == [1]  # 좌표로 찾은 기록만


async def test_other_users_comments_are_never_searched(parser):
    rows, details = _comment_world()
    others_rows = [make_row(9, comment="북한산 등산")]
    records = FakeMemoryRecords(
        {USER: details, 8: {9: make_record(9)}}, {USER: rows, 8: others_rows}
    )
    service = _service(parser, FakeVectorSearch(), records)

    res = await service.search(_request("가을에 북한산에서 들은 노래"))

    assert records.row_calls == [(USER, 5000)]  # 이 사용자의 자물쇠만 읽는다
    assert 9 not in [m.record_id for m in res.matched_records]
    assert "'북한산'은 아직 모르는 장소라서" in res.answer_text  # 다른 사람 코멘트로는 못 찾는다


async def test_questions_without_unknown_places_get_no_notice(parser):
    rows, details = _world()
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    assert "모르는 장소" not in res.answer_text
    assert "장소 목록에 없는" not in res.answer_text
