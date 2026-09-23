"""기능5 기억 검색 — 위키 모델 API 설계 7-1."""

from datetime import datetime

from pydantic import Field

from app.schemas.common import CamelModel, Place, Track


class MemorySearchRequest(CamelModel):
    user_id: str = Field(min_length=1)
    query_text: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1, le=20)


class MatchedRecord(CamelModel):
    record_id: str
    track: Track
    place: Place
    created_at: datetime
    similarity_score: float


class MemorySearchResponse(CamelModel):
    answer_text: str
    matched_records: list[MatchedRecord]
    retry_count: int = 0
