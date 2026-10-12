"""기능5 기억 검색 — 위키 모델 API 설계 7-1."""

from datetime import datetime

from pydantic import Field, field_validator

from app.schemas.common import CamelModel, Place, Track


class MemorySearchRequest(CamelModel):
    # 백엔드 userId는 숫자(Long). 숫자 문자열("35")도 받아서 숫자로 바꾼다.
    user_id: int = Field(gt=0)
    query_text: str = Field(min_length=1)
    # 백엔드 프론트 계약(POST /api/memories/search)은 records 최대 5개이고 limit을 따로 받지 않는다.
    # 그래서 기본값은 5다. 백엔드가 limit을 보내면 그 값(최대 20)을 따른다.
    limit: int = Field(default=5, ge=1, le=20)

    @field_validator("query_text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # min_length는 공백만 있는 질문을 못 거른다 — 임베딩이 빈 텍스트를 거부해 500이 된다
        stripped = value.strip()
        if not stripped:
            raise ValueError("질문이 비어 있어요.")
        return stripped


class MatchedRecord(CamelModel):
    record_id: int  # 공통 규칙: recordId는 숫자(Long). 프론트 계약도 integer int64
    track: Track
    place: Place
    created_at: datetime
    similarity_score: float


class MemorySearchResponse(CamelModel):
    answer_text: str
    matched_records: list[MatchedRecord]
    retry_count: int = 0
