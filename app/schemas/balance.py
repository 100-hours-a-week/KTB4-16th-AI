"""기능6 취향 투표 (클러스터·문항) — 위키 모델 API 설계 8-1 ~ 8-3."""

from typing import Literal

from pydantic import Field

from app.schemas.common import CamelModel


class QuestionGenerateRequest(CamelModel):
    cluster_id: str = Field(min_length=1)
    cluster_label: str = Field(min_length=1)
    count: int = Field(default=5, ge=1, le=20)


class BalanceQuestion(CamelModel):
    text: str
    option_yes: str
    option_no: str
    topic: str
    cluster_id: str


class QuestionGenerateResponse(CamelModel):
    questions: list[BalanceQuestion]


class ClusterAssignRequest(CamelModel):
    user_id: str = Field(min_length=1)


class ClusterAssignResponse(CamelModel):
    cluster_id: str | None
    cluster_label: str | None
    confidence: float | None


class ClusterRebuildQueuedResponse(CamelModel):
    status: Literal["QUEUED"] = "QUEUED"
    job_id: str
