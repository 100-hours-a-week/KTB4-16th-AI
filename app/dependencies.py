"""공통 의존성: 내부 인증, DB 세션, 외부 클라이언트 조립."""

import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache

from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.anthropic_client import AnthropicClient
from app.clients.clip_client import ClipClient, FakeClipClient, HttpClipClient
from app.clients.embedding_client import (
    EmbeddingClient,
    FakeEmbeddingClient,
    OpenAIEmbeddingClient,
)
from app.clients.llm_client import FakeLLMClient, LLMClient
from app.config import get_settings
from app.db.postgres import get_session
from app.db.repositories.job_repository import JobRepository
from app.exceptions import UnauthorizedError


async def verify_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    expected = get_settings().internal_token
    if not secrets.compare_digest(x_internal_token or "", expected):
        raise UnauthorizedError()


@dataclass(frozen=True)
class Clients:
    llm: LLMClient
    embedding: EmbeddingClient
    clip: ClipClient


@lru_cache
def get_clients() -> Clients:
    s = get_settings()
    if s.ai_client_mode == "fake":
        return Clients(
            llm=FakeLLMClient(),
            embedding=FakeEmbeddingClient(dim=s.text_embedding_dim),
            clip=FakeClipClient(dim=s.clip_embedding_dim),
        )
    return Clients(
        llm=AnthropicClient(
            api_key=s.anthropic_api_key,
            model=s.llm_model,
            timeout=s.external_timeout_seconds,
            max_retries=s.external_max_retries,
        ),
        embedding=OpenAIEmbeddingClient(
            api_key=s.openai_api_key,
            model=s.text_embedding_model,
            dim=s.text_embedding_dim,
            timeout=s.external_timeout_seconds,
            max_retries=s.external_max_retries,
        ),
        clip=HttpClipClient(
            api_url=s.clip_api_url,
            api_key=s.clip_api_key,
            model=s.clip_model,
            dim=s.clip_embedding_dim,
            timeout=s.external_timeout_seconds,
        ),
    )


async def get_job_repository(
    session: AsyncSession = Depends(get_session),
) -> AsyncIterator[JobRepository]:
    yield JobRepository(session)
