"""기능5 기억 검색 — 요청·응답 스키마 (AI 기억 검색 API 명세 3장).

응답은 ID 중심이다: records는 recordId만, suggestedRecords는 recordId와 matchedConditions만 담는다.
카드 정보(제목·가수·날짜·장소·날씨·이미지)는 백엔드가 recordId로 자기 DB에서 만든다.
"""

from pydantic import AliasChoices, Field, field_validator

from app.schemas.common import CamelModel


class MemorySearchRequest(CamelModel):
    # 백엔드 userId는 숫자(Long). 숫자 문자열("35")도 받아서 숫자로 바꾼다.
    user_id: int = Field(gt=0)
    # 백엔드 프론트 계약의 요청 필드는 query다. 예전 이름 queryText도 계속 받는다.
    query_text: str = Field(min_length=1, validation_alias=AliasChoices("query", "queryText"))
    # 프론트는 앞 5건만 채팅에 보이고 "전체보기"(임시 창)에서도 최대 50건까지만 보여주므로,
    # 조건에 맞는 자물쇠를 최대 50건 돌려준다. 백엔드가 limit을 보내면 그 값(1~50)을 따른다.
    limit: int = Field(default=50, ge=1, le=50)

    @field_validator("query_text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # min_length는 공백만 있는 질문을 못 거른다 — 임베딩이 빈 텍스트를 거부해 500이 된다
        stripped = value.strip()
        if not stripped:
            raise ValueError("질문이 비어 있어요.")
        return stripped


class MemoryRecordItem(CamelModel):
    """찾은 자물쇠 하나 — ID만 준다.

    제목·가수·날짜·장소·날씨·이미지 같은 카드 정보는 보내지 않는다. 백엔드가 recordId로 자기 DB에서
    카드를 만든다 (백엔드 제안, AI 동의). recordId는 요청 userId 소유이고 삭제되지 않은
    자물쇠만 담긴다.
    """

    record_id: int  # 공통 규칙: recordId는 숫자(Long)


class SuggestedRecordItem(MemoryRecordItem):
    """모든 조건에는 안 맞고 일부만 맞는 자물쇠. recordId와, 맞은 조건의 이름만 준다."""

    # 이 자물쇠가 맞은 조건의 이름 (예: ["서울숲 근처"]). 프론트가 "OO만 맞아요"로 보여준다.
    # 다시 계산할 수 없으니 백엔드가 대화를 저장할 때 같이 저장해야 한다
    matched_conditions: list[str]


class MemorySearchResponse(CamelModel):
    # 건수 숫자는 쓰지 않는다 (건수는 프론트가 records.length로 표시한다)
    answer: str
    # 조건에 모두 맞는 자물쇠 (점수 높은 순, 같으면 최근 순, 최대 50건).
    # 조건이 없는 질문은 비슷한 자물쇠
    records: list[MemoryRecordItem]
    # 일부만 맞는 자물쇠 (최대 3건, records가 5건보다 적을 때만). 없으면 빈 목록.
    # 프론트가 "이런 자물쇠를 찾으시는 건 아닌가요?" 토글을 누르면 카드로 보여준다.
    suggested_records: list[SuggestedRecordItem] = []
