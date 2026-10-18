"""조건이 있는 질문의 기억 검색 — 실제 규칙 파일(places/regions/question_rules)로 해석한다."""

from datetime import date, datetime

import pytest

from app.components.answer_headline import HeadlineWriter, build_prompt
from app.components.memory_answer import NO_RESULT_TEXT
from app.components.question_parser import QuestionParser
from app.schemas.memory_search import MemorySearchRequest
from app.services.memory_search_service import MemorySearchService
from tests.memory_fakes import (
    FakeMemoryRecords,
    FakeVectorSearch,
    StubEmbedder,
    StubLLM,
    hit,
    make_record,
    make_row,
)

USER = 7
TODAY = date(2026, 10, 12)


@pytest.fixture(scope="module")
def parser() -> QuestionParser:
    return QuestionParser.load_default()


def _service(parser, search, records, **overrides) -> MemorySearchService:
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
        **overrides,
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


async def test_exact_matches_are_records_and_partial_matches_are_suggested_records(parser):
    rows, details = _world()
    search = _search()
    records = FakeMemoryRecords({USER: details}, rows_by_user={USER: rows})
    service = _service(parser, search, records)

    res = await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    assert [m.record_id for m in res.records] == [1]  # 가을 AND 서울숲
    # 문장에는 앞머리만 있다. 목록도, 제안 문구도, 건수도 없다 (카드와 토글은 프론트가 그린다)
    assert res.answer == "서울숲 근처, 가을 조건에 맞는 자물쇠를 찾았어요."  # 건수 숫자 없음
    # 일부만 맞는 자물쇠는 suggestedRecords로 나가고, 어떤 조건만 맞았는지 같이 온다
    suggested = {s.record_id: s.matched_conditions for s in res.suggested_records}
    assert suggested == {2: ["서울숲 근처"], 3: ["가을"]}  # 4번(둘 다 안 맞음)은 나오지 않는다


async def test_records_found_only_by_conditions_get_their_similarity_looked_up(parser):
    rows, details = _world()
    search = _search()
    service = _service(parser, search, FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    assert search.similarity_calls == [(USER, [1, 2])]  # 이미 후보인 3·4번은 다시 구하지 않는다
    # 조건으로만 찾은 1번도 records에 들어간다 (점수는 응답에 담지 않는다 — 명세 3.2)
    assert [m.record_id for m in res.records] == [1]


async def test_suggestions_are_left_out_when_exact_matches_fill_the_limit(parser):
    rows, details = _world()
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래", limit=1))

    assert [m.record_id for m in res.records] == [1]
    assert res.suggested_records == []


async def test_no_exact_match_returns_empty_records_and_the_suggestions(parser):
    rows, details = _world()
    rows = [r for r in rows if r.record_id != 1]  # 가을×서울숲 기록이 없는 사용자
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래"))

    assert res.records == []
    assert res.answer == "서울숲 근처, 가을 조건에 모두 맞는 자물쇠는 없어요."
    assert {s.record_id for s in res.suggested_records} == {2, 3}


async def test_question_without_conditions_skips_the_condition_path(parser):
    rows, details = _world()
    records = FakeMemoryRecords({USER: details}, {USER: rows})
    search = _search()
    service = _service(parser, search, records)

    res = await service.search(_request("그때 들은 노래"))

    assert records.row_calls == []  # 자물쇠 정보를 읽지 않는다
    assert search.similarity_calls == [] and search.mood_calls == []
    assert res.answer.startswith("비슷한 자물쇠")  # 1단계 기준선 문장


async def test_dong_word_the_user_never_visited_falls_back_to_plain_search(parser):
    rows, details = _world()
    records = FakeMemoryRecords({USER: details}, {USER: rows})
    service = _service(parser, _search(), records)

    res = await service.search(_request("운동을 하던 날 들은 노래"))

    assert records.row_calls == [(USER, 5000)]  # 후보를 확인하려고 읽었지만
    assert res.answer.startswith("비슷한 자물쇠")  # 조건으로 남지 않아 기준선으로 처리
    assert [m.record_id for m in res.records] == [4, 3]


async def test_dong_the_user_has_is_a_condition_and_matches_by_name_prefix(parser):
    rows, details = _world()
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("성수동에서 들은 노래"))

    assert sorted(m.record_id for m in res.records) == [1, 2]  # 성수동1가·2가


async def test_region_without_any_records_gives_no_result_instead_of_unrelated_ones(parser):
    rows, details = _world()
    rows = [make_row(r.record_id, dong_code="1144010100") for r in rows]  # 모두 마포구
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("성동구에서 들은 노래"))

    assert res.records == []
    assert res.answer == NO_RESULT_TEXT


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

    assert sorted(m.record_id for m in res.records) == [1, 2]


async def test_relative_period_uses_todays_date(parser):
    rows = [
        make_row(1, created_at=datetime(2026, 9, 10, 9, 0)),
        make_row(2, created_at=datetime(2026, 10, 1, 9, 0)),
        make_row(3, created_at=datetime(2026, 9, 30, 23, 0)),
    ]
    details = {i: make_record(i) for i in (1, 2, 3)}
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("지난달에 들은 노래"))  # 오늘 2026-10-12 → 9월

    assert sorted(m.record_id for m in res.records) == [1, 3]


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
    assert sorted(m.record_id for m in res.records) == [1, 2]


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
    assert res.records == []
    assert res.answer == NO_RESULT_TEXT


async def test_exact_record_that_no_longer_exists_is_skipped(parser):
    rows, details = _world()
    del details[1]  # AI DB에는 남아 있지만 백엔드에서 삭제된 자물쇠
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래"))

    assert res.records == []
    assert res.answer.startswith("서울숲 근처, 가을 조건에 모두 맞는 자물쇠는 없어요.")


async def test_combined_period_place_mood_question(parser):
    rows = [
        make_row(1, created_at=datetime(2025, 10, 3), mood_score=-30, **SEOUL_FOREST),
        make_row(2, created_at=datetime(2025, 10, 4), mood_score=30, **SEOUL_FOREST),  # 기분 좋음
        make_row(3, created_at=datetime(2026, 10, 3), mood_score=-30, **SEOUL_FOREST),  # 올해
    ]
    details = {i: make_record(i) for i in (1, 2, 3)}
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("작년 가을 서울숲에서 우울했을 때 들은 노래"))

    assert [m.record_id for m in res.records] == [1]


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

    assert [m.record_id for m in res.records] == [1]  # 가을 AND 코멘트에 광교산
    suggested = {s.record_id: s.matched_conditions for s in res.suggested_records}
    assert suggested[2] == ["광교산"]  # 2번(봄)은 장소만 맞아서 제안으로
    assert suggested[3] == suggested[4] == ["가을"]  # 3·4번(가을)은 계절만 맞아서 제안으로
    assert res.answer.endswith(
        "'광교산'은 장소 목록에 없는 곳이라서, 기록에 적어 둔 코멘트에서 찾았어요."
    )


async def test_unknown_place_nobody_wrote_about_is_ignored_with_a_notice(parser):
    rows, details = _comment_world()
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 북한산에서 들은 노래"))

    # 위치 조건은 빠지고 가을 조건만으로 찾는다
    assert sorted(m.record_id for m in res.records) == [1, 3, 4]
    assert res.answer.startswith("가을 조건에 맞는 자물쇠를 찾았어요.")
    assert res.answer.endswith(
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

    assert [m.record_id for m in res.records] == [3]  # 벡터 검색 결과 그대로
    assert res.answer.startswith("비슷한 자물쇠를 찾았어요.")
    assert "'북한산'은 아직 모르는 장소라서" in res.answer


async def test_notice_is_added_even_when_nothing_is_found(parser):
    rows, details = _comment_world()
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("북한산에서 들은 노래"))

    assert res.records == []
    assert res.answer.startswith(NO_RESULT_TEXT)
    assert "'북한산'은 아직 모르는 장소라서" in res.answer


async def test_known_place_without_nearby_records_is_found_through_the_comment(parser):
    far = {"latitude": 37.5575, "longitude": 126.9248}  # 홍대 — 서울숲에서 멀다
    rows = [
        make_row(1, **far, comment="서울숲 산책하다 들은 곡"),
        make_row(2, **far, comment="합정에서 약속"),
    ]
    details = {i: make_record(i) for i in (1, 2)}
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("서울숲에서 들은 노래"))

    assert [m.record_id for m in res.records] == [1]
    assert "장소 목록에 없는" not in res.answer  # 사전에 있는 장소라 안내할 것이 없다


async def test_known_place_with_nearby_records_ignores_comment_mentions_elsewhere(parser):
    rows = [
        make_row(1, **SEOUL_FOREST, comment=None),
        make_row(2, **HONGDAE, comment="서울숲에 다녀온 날"),
    ]
    details = {i: make_record(i) for i in (1, 2)}
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("서울숲에서 들은 노래"))

    assert [m.record_id for m in res.records] == [1]  # 좌표로 찾은 기록만


async def test_other_users_comments_are_never_searched(parser):
    rows, details = _comment_world()
    others_rows = [make_row(9, comment="북한산 등산")]
    records = FakeMemoryRecords(
        {USER: details, 8: {9: make_record(9)}}, {USER: rows, 8: others_rows}
    )
    service = _service(parser, FakeVectorSearch(), records)

    res = await service.search(_request("가을에 북한산에서 들은 노래"))

    assert records.row_calls == [(USER, 5000)]  # 이 사용자의 자물쇠만 읽는다
    assert 9 not in [m.record_id for m in res.records]
    assert "'북한산'은 아직 모르는 장소라서" in res.answer  # 다른 사람 코멘트로는 못 찾는다


async def test_questions_without_unknown_places_get_no_notice(parser):
    rows, details = _world()
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    assert "모르는 장소" not in res.answer
    assert "장소 목록에 없는" not in res.answer


# ---- 전체 반환: 화면에는 5건, 나머지는 "더보기" ------------------------------


def _forest_world(autumn: int, spring: int = 0):
    """서울숲에서 가을에 저장한 기록 autumn건, 봄에 저장한 기록 spring건을 가진 사용자."""
    rows, details = [], {}
    for i in range(1, autumn + spring + 1):
        when = datetime(2026, 10, 3, 9, 0) if i <= autumn else datetime(2026, 5, 2, 9, 0)
        rows.append(make_row(i, created_at=when, **SEOUL_FOREST))
        details[i] = make_record(i, created_at=when, place_name="성수동1가")
    return rows, details


async def test_all_exact_matches_are_returned_even_beyond_the_five_shown_first(parser):
    rows, details = _forest_world(autumn=6, spring=2)
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들은 노래"))

    assert sorted(m.record_id for m in res.records) == [1, 2, 3, 4, 5, 6]
    # 화면에 먼저 보이는 5건이 찼으므로 일부만 맞는 기록(7·8번)은 제안하지 않는다
    assert res.answer == "서울숲 근처, 가을 조건에 맞는 자물쇠를 찾았어요."


async def test_suggestions_are_added_while_fewer_than_five_exact_matches(parser):
    rows, details = _forest_world(autumn=4, spring=2)
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들은 노래"))

    assert len(res.records) == 4
    assert res.answer.startswith("서울숲 근처, 가을 조건에 맞는 자물쇠를 찾았어요.")
    assert [s.matched_conditions for s in res.suggested_records] == [["서울숲 근처"]] * 2  # 5·6번


async def test_headline_says_only_some_are_shown_when_fewer_are_returned_than_found(parser):
    rows, details = _forest_world(autumn=5)
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들은 노래", limit=3))

    assert len(res.records) == 3
    # 건수 숫자는 쓰지 않는다. 몇 건인지는 프론트가 records.length로 센다
    assert res.answer == "서울숲 근처, 가을 조건에 맞는 자물쇠가 많아서 일부만 보여드려요."


async def test_exact_matches_are_returned_in_score_order_and_the_list_is_not_in_the_text(parser):
    rows, details = _forest_world(autumn=3)
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들은 노래"))

    assert [m.record_id for m in res.records] == [3, 2, 1]  # 점수가 같으면 최근 기록 먼저
    assert "밤편지" not in res.answer and "성수동1가" not in res.answer


# ---- 앞머리 문장을 LLM이 쓸 때 -------------------------------------------------

LLM_SENTENCE = "서울숲 근처에서 가을에 저장하신 자물쇠를 찾았습니다."


def _llm_service(parser, llm, records, search=None, **overrides):
    return _service(
        parser,
        search or _search(),
        records,
        headline_writer=HeadlineWriter(llm),
        **overrides,
    )


async def test_headline_comes_from_the_llm_when_it_is_turned_on(parser):
    rows, details = _world()
    llm = StubLLM(LLM_SENTENCE)
    service = _llm_service(parser, llm, FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    assert res.answer.splitlines()[0] == LLM_SENTENCE
    assert [m.record_id for m in res.records] == [1]  # 결과는 LLM과 상관없이 검색 그대로
    # 일부만 맞는 자물쇠는 문장이 아니라 suggestedRecords로 나간다 (LLM과 상관없이 검색 그대로)
    assert {s.record_id for s in res.suggested_records} == {2, 3}
    assert "이런 자물쇠" not in res.answer


async def test_llm_receives_only_condition_names_and_whether_there_are_results(parser):
    rows, details = _world()
    llm = StubLLM(LLM_SENTENCE)
    service = _llm_service(parser, llm, FakeMemoryRecords({USER: details}, {USER: rows}))

    await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    (call,) = llm.calls
    assert call["prompt"] == build_prompt(["서울숲 근처", "가을"], found=1, total=1, suggestions=2)
    assert not any(ch.isdigit() for ch in call["prompt"])  # 건수도 보내지 않는다
    for private in ("들었던 노래", "밤편지", "아이유", "성수동", "반포동"):
        assert private not in call["prompt"]  # 질문 원문·곡·장소는 보내지 않는다


async def test_llm_failure_falls_back_to_the_template_headline(parser):
    rows, details = _world()
    llm = StubLLM(error=RuntimeError("LLM 서버 오류"))
    service = _llm_service(parser, llm, FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    assert res.answer.splitlines()[0] == "서울숲 근처, 가을 조건에 맞는 자물쇠를 찾았어요."
    assert [m.record_id for m in res.records] == [1]  # 검색 결과는 그대로


async def test_llm_sentence_that_states_a_count_is_not_used(parser):
    rows, details = _world()
    llm = StubLLM("서울숲 근처에서 가을에 저장하신 자물쇠 1건을 찾았습니다.")
    service = _llm_service(parser, llm, FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    assert res.answer.splitlines()[0] == "서울숲 근처, 가을 조건에 맞는 자물쇠를 찾았어요."


async def test_slow_llm_is_given_up_after_the_timeout(parser):
    rows, details = _world()
    llm = StubLLM(LLM_SENTENCE, delay=1.0)
    service = _llm_service(
        parser,
        llm,
        FakeMemoryRecords({USER: details}, {USER: rows}),
        headline_timeout=0.01,
    )

    res = await service.search(_request("가을에 서울숲에서 들었던 노래 뭐지?"))

    assert res.answer.splitlines()[0] == "서울숲 근처, 가을 조건에 맞는 자물쇠를 찾았어요."


async def test_llm_is_not_called_when_there_is_nothing_to_report(parser):
    rows, details = _world()
    llm = StubLLM(LLM_SENTENCE)
    records = FakeMemoryRecords({USER: details}, {USER: rows})
    service = _llm_service(parser, llm, records)

    res = await service.search(_request("가을에 서울숲에서 들었던 노래", userId=999))

    assert res.answer == NO_RESULT_TEXT
    assert llm.calls == []  # 근거가 없는데 문장을 만들게 하지 않는다


async def test_llm_is_not_called_for_questions_without_conditions(parser):
    rows, details = _world()
    llm = StubLLM(LLM_SENTENCE)
    service = _llm_service(parser, llm, FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("그때 들은 노래"))

    assert llm.calls == []  # 나열할 조건이 없다
    assert res.answer.startswith("비슷한 자물쇠")


async def test_llm_headline_is_followed_by_the_notice_written_by_code(parser):
    rows, details = _comment_world()
    llm = StubLLM("가을에 저장하신 자물쇠를 찾았어요.")
    service = _llm_service(
        parser,
        llm,
        FakeMemoryRecords({USER: details}, {USER: rows}),
        search=FakeVectorSearch(),
    )

    res = await service.search(_request("가을에 북한산에서 들은 노래"))

    # 어느 코멘트에도 없는 북한산은 조건에서 빠져서 LLM에는 "가을"만 전달된다
    assert llm.calls[0]["prompt"] == build_prompt(["가을"], found=3, total=3, suggestions=0)
    lines = res.answer.splitlines()
    assert lines[0] == "가을에 저장하신 자물쇠를 찾았어요."
    assert lines[-1].startswith("'북한산'은 아직 모르는 장소라서 위치 조건은 적용하지 못했어요.")


# ---- 제안(suggestedRecords): 토글을 열면 보이는 최대 3건 ----------------------


async def test_suggested_records_are_limited_to_three(parser):
    rows, details = _forest_world(autumn=1, spring=5)  # 가을×서울숲 1건, 봄×서울숲 5건(일부만 맞음)
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들은 노래"))

    assert len(res.records) == 1
    assert len(res.suggested_records) == 3  # 5건이 후보여도 3건까지만


async def test_suggested_records_are_empty_when_exact_matches_fill_the_screen(parser):
    rows, details = _forest_world(autumn=6, spring=2)
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들은 노래"))

    assert len(res.records) == 6
    assert res.suggested_records == []  # 채팅에 보이는 5건이 찼으면 제안하지 않는다


async def test_suggested_record_is_the_id_with_the_conditions_it_matched(parser):
    rows, details = _forest_world(autumn=1, spring=1)
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들은 노래"))

    (suggested,) = res.suggested_records
    # 카드 정보는 백엔드가 recordId로 만든다 — 여기서는 ID와 맞은 조건 이름뿐이다
    assert suggested.model_dump(by_alias=True) == {
        "recordId": 2,
        "matchedConditions": ["서울숲 근처"],  # 봄이라 가을은 안 맞음
    }


async def test_suggested_record_that_no_longer_exists_is_skipped(parser):
    rows, details = _forest_world(autumn=1, spring=2)
    del details[2]  # 백엔드에서 삭제된 자물쇠 (records 테이블에 없음)
    service = _service(parser, FakeVectorSearch(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("가을에 서울숲에서 들은 노래"))

    assert [s.record_id for s in res.suggested_records] == [3]


async def test_no_suggested_records_for_questions_without_conditions(parser):
    rows, details = _world()
    service = _service(parser, _search(), FakeMemoryRecords({USER: details}, {USER: rows}))

    res = await service.search(_request("그때 들은 노래"))

    assert res.suggested_records == []
