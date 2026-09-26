"""기능1 상황 맞춤 추천·플레이리스트 저장 — 위키 모델 API 설계 3절."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel, Place, Track, Weather


class ContextRecommendRequest(CamelModel):
    request_id: str = Field(min_length=1)
    # 백엔드 명세(MULO_API설계): 외부 API에서도 내부 user_id(Long)를 그대로 쓴다
    user_id: int = Field(gt=0)
    place: Place
    weather: Weather
    local_time: datetime
    limit: int = Field(default=3, ge=1, le=10)


class RankedTrack(Track):
    rerank_score: float


class ContextRecommendResponse(CamelModel):
    request_id: str
    rewritten_query: str
    recommendation_basis: Literal["PERSONAL", "REGIONAL", "GENERIC"]
    tracks: list[RankedTrack]
    degraded: bool = False


class PlaylistSaveRequest(CamelModel):
    user_id: int = Field(gt=0)
    track_uris: list[str] = Field(min_length=1)
    platform: Literal["spotify", "youtube"]
    title: str = Field(min_length=1)


class PlaylistSaveResponse(CamelModel):
    playlist_url: str
    platform: Literal["spotify", "youtube"]
