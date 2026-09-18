"""기능4 임베딩 인덱싱 — 필드 이름·타입은 백엔드 명세(MULO_API설계)를 따른다.

백엔드가 POST /api/records 커밋 후, 임베딩에 필요한 필드만 골라 호출한다.
"""

from datetime import datetime

from pydantic import Field, field_validator

from app.schemas.common import CamelModel


class EmbeddingTrack(CamelModel):
    title: str = Field(min_length=1)
    artist_name: str = Field(min_length=1)
    # 스포티파이 곡 ID — 곡별 무드 묘사를 재사용하는 키. 장르·오디오 특성은 스포티파이가 주지 않는다
    external_track_id: str = Field(min_length=1)

    @field_validator("external_track_id")
    @classmethod
    def _strip_uri_prefix(cls, value: str) -> str:
        # "spotify:track:xxx" 형태로 와도 ID만 남긴다
        return value.rsplit(":", 1)[-1]


class EmbeddingGenerateRequest(CamelModel):
    record_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    # 자물쇠 사진은 필수
    photo_url: str = Field(min_length=1)
    track: EmbeddingTrack
    # null은 정상 상태 — 코멘트 임베딩 단계만 생략
    comment: str | None = None
    created_at: datetime
