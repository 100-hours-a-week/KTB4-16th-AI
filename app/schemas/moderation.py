"""기능6 모더레이션 — 위키 모델 API 설계 8-4."""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel


class ModerationCheckRequest(CamelModel):
    request_id: str = Field(min_length=1)
    room_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=500)


class ModerationCheckResponse(CamelModel):
    request_id: str
    is_toxic: bool
    confidence: float
    category: Literal["HATE", "ABUSE", "SPAM"] | None
    model_version: str
