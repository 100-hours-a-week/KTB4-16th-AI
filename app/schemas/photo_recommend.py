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
    # 규칙 기반 검색어 완화로 재검색한 횟수. 0이면 첫 시도에 성공 (기능5 retryCount와 같은 규칙)
    retry_count: int = 0
