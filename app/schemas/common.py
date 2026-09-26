"""공통 타입 — 백엔드 문서 5.1절과 동일 (camelCase 직렬화)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Track(CamelModel):
    title: str
    artist: str
    # 백엔드는 곡을 Spotify 곡 ID(music_tracks.external_track_id)로 식별한다
    external_track_id: str
    spotify_uri: str | None = None
    album_art_url: str | None = None
    # 백엔드 music_tracks.external_url (open.spotify.com 링크)
    external_url: str | None = None


class Place(CamelModel):
    # 백엔드 명세(MULO_API설계)상 placeId는 Long
    place_id: int
    # 백엔드 places엔 법정동 이름(legal_dong_name)만 있고, 그마저 비어 있을 수 있다
    name: str | None = None
    lat: float | None = None
    lng: float | None = None


class Weather(CamelModel):
    # 백엔드 GET /weather/current 응답값 그대로 (형식 고정 안 함)
    condition: str
    temperature: float | None = None


class Mood(CamelModel):
    score: int = Field(ge=1, le=5)


class QueuedResponse(CamelModel):
    status: Literal["QUEUED"] = "QUEUED"
    job_id: str
