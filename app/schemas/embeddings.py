"""기능4 임베딩 인덱싱 — 필드 이름·타입은 백엔드 명세(MULO_API설계)를 따른다.

백엔드가 POST /api/records 커밋 후, 임베딩에 필요한 필드만 골라 호출한다.
"""

from datetime import datetime

from pydantic import Field

from app.schemas.common import CamelModel


class EmbeddingTrack(CamelModel):
    title: str = Field(min_length=1)
    artist_name: str = Field(min_length=1)
    # 무드 태깅 프롬프트에 넣어 LLM이 모르는 곡의 추정 정확도를 높인다
    genre: str | None = None
    external_track_id: str | None = None


class EmbeddingGenerateRequest(CamelModel):
    record_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    # 자물쇠 사진은 필수
    photo_url: str = Field(min_length=1)
    track: EmbeddingTrack
    # null은 정상 상태 — 코멘트 임베딩 단계만 생략
    comment: str | None = None
    created_at: datetime
