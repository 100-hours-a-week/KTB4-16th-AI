"""기능1 상황 맞춤 추천·플레이리스트 저장 — 위키 모델 API 설계 3절."""

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel, Place, Track, Weather


class ContextRecommendRequest(CamelModel):
    # 백엔드 추천 플레이리스트 생성 흐름은 요청 ID를 따로 만들지 않는다 — 오면 응답에 그대로 돌려줌
    request_id: str | None = None
    # 백엔드 명세(MULO_API설계): 외부 API에서도 내부 user_id(Long)를 그대로 쓴다
    user_id: int = Field(gt=0)
    # 백엔드는 날씨 조회에만 좌표를 쓰고 장소는 안 넘긴다. 오면 장소 이름을 프롬프트에 넣는다
    place: Place | None = None
    weather: Weather
    local_time: datetime
    limit: int = Field(default=3, ge=1, le=10)


class RankedTrack(Track):
    rerank_score: float


class ContextRecommendResponse(CamelModel):
    request_id: str | None = None
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
