"""공통 타입 — 백엔드 문서 5.1절과 동일 (camelCase 직렬화)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Track(CamelModel):
    title: str
    artist: str
    spotify_uri: str | None = None
    album_art_url: str | None = None


class Place(CamelModel):
    place_id: str
    name: str
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
