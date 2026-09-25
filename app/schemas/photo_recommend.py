"""기능3 사진 기반 추천 — 위키 모델 API 설계 5-1."""

from pydantic import Field

from app.schemas.common import CamelModel, Place, Track


class PhotoRecommendRequest(CamelModel):
    image_url: str = Field(min_length=1)
    place: Place | None = None


class PhotoRecommendResponse(CamelModel):
    mood_tags: list[str]
    # 장르 키워드 검색을 LLM 곡 추천으로 바꾸면서 생성되는 검색어가 없어졌다.
    # 백엔드 계약 유지를 위해 필드는 남기고 빈 문자열을 보낸다 (스펙 정리 시 제거 예정).
    rewritten_query: str
    tracks: list[Track]
    degraded: bool = False
    # 검색어 완화 재시도가 사라져 항상 0. 위 rewritten_query와 같은 이유로 남겨둔 필드.
    retry_count: int = 0
