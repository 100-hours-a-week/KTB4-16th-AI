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
from app.clients.spotify_client import (
    FakeSpotifyPlaylistClient,
    FakeSpotifySearchClient,
    SpotifyClient,
    SpotifyPlaylistClient,
    SpotifySearchClient,
    SpotifyServiceAccountClient,
)
from app.components.category_vectors import CategoryVectors
from app.components.clip_tagger import ClipTagger
from app.components.embedder import Embedder
from app.components.query_rewriter import QueryRewriter
from app.components.reranker import Reranker
from app.config import get_settings
from app.db.postgres import get_session
from app.db.repositories.embedding_repository import EmbeddingRepository, EmbeddingStore
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.track_mood_repository import TrackMoodRepository
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
    spotify_search: SpotifySearchClient
    spotify_playlist: SpotifyPlaylistClient


@lru_cache
def get_clients() -> Clients:
    s = get_settings()
    if s.ai_client_mode == "fake":
        return Clients(
            llm_knowledge=FakeLLMClient(),
            llm_general=FakeLLMClient(),
            embedding=FakeEmbeddingClient(dim=s.text_embedding_dim),
            clip=FakeClipClient(dim=s.clip_embedding_dim),
            spotify_search=FakeSpotifySearchClient(),
            spotify_playlist=FakeSpotifyPlaylistClient(),
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
        spotify_search=SpotifyClient(
            client_id=s.spotify_client_id,
            client_secret=s.spotify_client_secret,
            timeout=s.external_timeout_seconds,
        ),
        spotify_playlist=SpotifyServiceAccountClient(
            client_id=s.spotify_client_id,
            client_secret=s.spotify_client_secret,
            refresh_token=s.spotify_service_refresh_token,
            timeout=s.external_timeout_seconds,
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


def get_spotify_playlist_client() -> SpotifyPlaylistClient:
    return get_clients().spotify_playlist


def get_spotify_search_client() -> SpotifySearchClient:
    return get_clients().spotify_search


@lru_cache
def get_category_vectors() -> CategoryVectors:
    """태그 벡터 캐시는 프로세스당 1개. 첫 호출 때 CategoryVectors.load()가 CLIP을
    부르고, 그 뒤로는 재사용된다."""
    return CategoryVectors(get_clients().clip)


def get_clip_tagger() -> ClipTagger:
    return ClipTagger(get_clients().clip, get_category_vectors())


def get_query_rewriter() -> QueryRewriter:
    # 곡 지식이 필요 없는 언어 변환 작업이라 general LLM을 쓴다
    return QueryRewriter(get_clients().llm_general)


async def get_reranker(session: AsyncSession = Depends(get_session)) -> AsyncIterator[Reranker]:
    clients = get_clients()
    yield Reranker(
        embedder=Embedder(clients.embedding),
        track_moods=TrackMoodRepository(session),
        jobs=JobRepository(session),
    )
