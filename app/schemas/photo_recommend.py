"""기능3 사진 기반 추천 — 위키 모델 API 설계 5-1."""

from pydantic import Field

from app.schemas.common import CamelModel, Place, Track


class PhotoRecommendRequest(CamelModel):
    image_url: str = Field(min_length=1)
    place: Place | None = None


class PhotoRecommendResponse(CamelModel):
    mood_tags: list[str]
    rewritten_query: str
    tracks: list[Track]
    degraded: bool = False
