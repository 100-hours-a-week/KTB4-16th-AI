"""공통 의존성: 내부 인증, DB 세션, 외부 클라이언트 조립."""

import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.anthropic_client import AnthropicClient
from app.clients.clip_client import ClipClient, FakeClipClient, ReplicateClipClient
from app.clients.embedding_client import (
    EmbeddingClient,
    FakeEmbeddingClient,
    OpenAIEmbeddingClient,
)
from app.clients.llm_client import FakeLLMClient, LLMClient, LocalLLMClient
from app.config import get_settings
from app.db.postgres import get_session
from app.db.repositories.embedding_repository import EmbeddingRepository, EmbeddingStore
from app.db.repositories.job_repository import JobRepository
from app.exceptions import UnauthorizedError


async def verify_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    expected = get_settings().internal_token
    if not secrets.compare_digest(x_internal_token or "", expected):
        raise UnauthorizedError()


@dataclass(frozen=True)
class Clients:
    # 곡·가수 지식이 필요한 작업 (음악 무드 묘사, 투표 문항 생성)
    llm_knowledge: LLMClient
    # 언어 변환만 필요한 작업 (리라이팅, 요약, 답변, 클러스터 이름)
    llm_general: LLMClient
    embedding: EmbeddingClient
    clip: ClipClient


@lru_cache
def get_clients() -> Clients:
    s = get_settings()
    if s.ai_client_mode == "fake":
        return Clients(
            llm_knowledge=FakeLLMClient(),
            llm_general=FakeLLMClient(),
            embedding=FakeEmbeddingClient(dim=s.text_embedding_dim),
            clip=FakeClipClient(dim=s.clip_embedding_dim),
        )
    return Clients(
        llm_knowledge=_build_llm(s.llm_knowledge_provider),
        llm_general=_build_llm(s.llm_general_provider),
        embedding=OpenAIEmbeddingClient(
            api_key=s.openai_api_key,
            model=s.text_embedding_model,
            dim=s.text_embedding_dim,
            timeout=s.external_timeout_seconds,
            max_retries=s.external_max_retries,
        ),
        clip=ReplicateClipClient(
            api_token=s.replicate_api_token,
            model=s.clip_model,
            dim=s.clip_embedding_dim,
            timeout=s.external_timeout_seconds,
            max_retries=s.external_max_retries,
        ),
    )


def _build_llm(provider: str) -> LLMClient:
    s = get_settings()
    if provider == "local":
        return LocalLLMClient(url=s.local_llm_url, model=s.local_llm_model)
    return AnthropicClient(
        api_key=s.anthropic_api_key,
        model=s.llm_model,
        timeout=s.external_timeout_seconds,
        max_retries=s.external_max_retries,
    )


async def get_job_repository(
    session: AsyncSession = Depends(get_session),
) -> AsyncIterator[JobRepository]:
    yield JobRepository(session)


async def get_embedding_store(
    session: AsyncSession = Depends(get_session),
) -> AsyncIterator[EmbeddingStore]:
    yield EmbeddingRepository(session)
