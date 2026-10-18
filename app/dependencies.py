"""공통 의존성: 내부 인증, DB 세션, 외부 클라이언트 조립."""

import logging
import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, Header
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.clients.anthropic_client import AnthropicClient
from app.clients.backend_client import BackendClient, FakeBackendClient, ReportBackendClient
from app.clients.clip_client import ClipClient, FakeClipClient, ReplicateClipClient
from app.clients.embedding_client import (
    EmbeddingClient,
    FakeEmbeddingClient,
    LocalEmbeddingClient,
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
from app.components.answer_headline import HeadlineWriter
from app.components.axis_vectors import AxisVectors
from app.components.category_vectors import CategoryVectors
from app.components.clip_tagger import ClipTagger
from app.components.condition_scoring import ConditionWeights
from app.components.embedder import Embedder
from app.components.kure_model import FakeKureModel, TextEmbeddingModel, load_kure_model
from app.components.moderation_model import (
    FakeModerationModel,
    ModerationModel,
    load_onnx_model,
)
from app.components.query_rewriter import QueryRewriter
from app.components.question_generator import QuestionGenerator
from app.components.question_parser import QuestionParser
from app.components.reranker import Reranker
from app.components.saved_song_pool import SavedSongPool
from app.components.song_curator import SongCurator
from app.components.song_pool import SongPool, load_song_pool
from app.components.tag_vector_file import PrecomputedVectors, read_vectors_file
from app.components.tag_version import compute_tag_version
from app.components.vector_search import MemoryVectorSearch, RecordSearch, VectorSearch
from app.config import Settings, get_settings
from app.db import mysql, postgres
from app.db.postgres import get_session
from app.db.repositories.balance_stats_repository import (
    BalanceStatsRepository,
    GenreStats,
    GenreStatsSource,
)
from app.db.repositories.embedding_repository import EmbeddingRepository, EmbeddingStore
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.memory_record_repository import (
    ConditionRow,
    MemoryRecord,
    MemoryRecordRepository,
    MemoryRecordSource,
)
from app.db.repositories.report_repository import MonthlyReportRepository
from app.db.repositories.track_lookup_repository import TrackLookupRepository, TrackLookupStore
from app.db.repositories.track_mood_repository import TrackMoodRepository
from app.exceptions import UnauthorizedError, UpstreamError
from app.services.balance_service import BalanceService
from app.services.memory_search_service import MemorySearchService
from app.services.moderation_service import ModerationService


async def verify_internal_token(x_internal_token: str | None = Header(default=None)) -> None:
    expected = get_settings().internal_token
    if not secrets.compare_digest(x_internal_token or "", expected):
        raise UnauthorizedError()


logger = logging.getLogger("muro.dependencies")


@dataclass(frozen=True)
class Clients:
    # 곡·가수 지식이 필요한 작업 (음악 무드 묘사, 투표 문항 생성)
    llm_knowledge: LLMClient
    # 언어 변환만 필요한 작업 (리라이팅, 요약, 답변, 클러스터 이름)
    llm_general: LLMClient
    embedding: EmbeddingClient
    clip: ClipClient
    backend: ReportBackendClient
    spotify_search: SpotifySearchClient
    spotify_playlist: SpotifyPlaylistClient


def _build_embedding_client(s: Settings) -> EmbeddingClient:
    if s.text_embedding_provider == "openai":
        return OpenAIEmbeddingClient(
            api_key=s.openai_api_key,
            model=s.text_embedding_model,
            dim=s.text_embedding_dim,
            timeout=s.external_timeout_seconds,
            max_retries=s.external_max_retries,
        )
    return LocalEmbeddingClient(
        base_url=s.embedding_service_url,
        token=s.internal_token,
        model=s.kure_model_name,
        dim=s.kure_embedding_dim,
        timeout=s.external_timeout_seconds,
    )


@lru_cache
def get_clients() -> Clients:
    s = get_settings()
    if s.ai_client_mode == "fake":
        return Clients(
            llm_knowledge=FakeLLMClient(),
            llm_general=FakeLLMClient(),
            embedding=FakeEmbeddingClient(dim=s.text_vector_dim),
            clip=FakeClipClient(dim=s.clip_embedding_dim),
            backend=FakeBackendClient(),
            spotify_search=FakeSpotifySearchClient(),
            spotify_playlist=FakeSpotifyPlaylistClient(),
        )
    return Clients(
        llm_knowledge=_build_llm(s.llm_knowledge_provider),
        llm_general=_build_llm(s.llm_general_provider),
        embedding=_build_embedding_client(s),
        clip=ReplicateClipClient(
            api_token=s.replicate_api_token,
            model=s.clip_model,
            dim=s.clip_embedding_dim,
            timeout=s.external_timeout_seconds,
            max_retries=s.external_max_retries,
        ),
        backend=BackendClient(
            base_url=s.backend_internal_url,
            internal_token=s.internal_token,
            timeout=s.external_timeout_seconds,
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


async def get_monthly_report_repository(
    session: AsyncSession = Depends(get_session),
) -> AsyncIterator[MonthlyReportRepository]:
    yield MonthlyReportRepository(session)


async def get_track_lookup_store(
    session: AsyncSession = Depends(get_session),
) -> AsyncIterator[TrackLookupStore]:
    yield TrackLookupRepository(session)


async def get_saved_song_pool(
    session: AsyncSession = Depends(get_session),
) -> AsyncIterator[SavedSongPool]:
    """사용자들이 저장한 곡(track_moods)을 추천 후보로 꺼내 주는 두 번째 곡 풀."""
    yield SavedSongPool(TrackMoodRepository(session))


def get_spotify_playlist_client() -> SpotifyPlaylistClient:
    return get_clients().spotify_playlist


def get_spotify_search_client() -> SpotifySearchClient:
    return get_clients().spotify_search


@lru_cache
def get_song_pool() -> SongPool | None:
    return load_song_pool(Path(get_settings().song_pool_path))


@lru_cache
def get_precomputed_vectors() -> PrecomputedVectors | None:
    """미리 계산해 둔 태그 벡터(app/tag_vectors.json). 쓸 수 없으면 None.

    파일이 없거나, 지금의 태그 정의·CLIP 모델과 버전이 다르면 쓰지 않는다. 그 경우 아래
    CategoryVectors·AxisVectors가 첫 사용 때 CLIP으로 직접 계산하는데, 몇 분 걸린다.
    """
    vectors = read_vectors_file()
    if vectors is None:
        logger.warning("미리 계산한 태그 벡터 파일(app/tag_vectors.json)이 없거나 깨져 있어요.")
        return None
    expected = compute_tag_version(get_clients().clip.model)
    if vectors.version != expected:
        logger.warning(
            "미리 계산한 태그 벡터의 버전(%s)이 지금 태그 정의·CLIP 모델(%s, 버전 %s)과 달라서 "
            "쓰지 않아요. scripts/build_tag_vectors.py로 다시 만드세요.",
            vectors.version,
            get_clients().clip.model,
            expected,
        )
        return None
    return vectors


@lru_cache
def get_category_vectors() -> CategoryVectors:
    """태그 벡터 캐시는 프로세스당 1개. 미리 계산한 파일이 있으면 그걸 읽고, 없으면 첫 호출 때
    CategoryVectors.load()가 CLIP을 부른다. 그 뒤로는 재사용된다."""
    return CategoryVectors(get_clients().clip, precomputed=get_precomputed_vectors())


@lru_cache
def get_axis_vectors() -> AxisVectors:
    """다축 태그 벡터 캐시도 프로세스당 1개. 미리 계산한 파일이 있으면 그걸 읽고, 없으면 첫 사용 때
    태그 문장을 CLIP으로 인코딩한다."""
    return AxisVectors(get_clients().clip, precomputed=get_precomputed_vectors())


@lru_cache
def get_tag_version() -> str:
    """저장하는 사진 태그가 어떤 태그 목록·CLIP 모델로 분류됐는지 알려 주는 값."""
    return compute_tag_version(get_clients().clip.model)


def get_clip_tagger() -> ClipTagger:
    return ClipTagger(get_clients().clip, get_category_vectors(), get_axis_vectors())


def get_query_rewriter() -> QueryRewriter:
    # 곡 지식이 필요 없는 언어 변환 작업이라 general LLM을 쓴다
    return QueryRewriter(get_clients().llm_general)


def get_song_curator() -> SongCurator:
    # 실제 존재하는 곡·가수를 지목해야 하므로 곡 지식이 있는 LLM을 쓴다
    return SongCurator(get_clients().llm_knowledge)


def get_reranker() -> Reranker:
    # 태그 겹침 계산만 하는 순수 함수라 DB·외부 API 의존이 없다
    return Reranker()


def get_embedder() -> Embedder:
    return Embedder(get_clients().embedding)


async def get_record_search(
    session: AsyncSession = Depends(get_session),
) -> AsyncIterator[RecordSearch]:
    yield VectorSearch(session)


async def get_memory_vector_search(
    session: AsyncSession = Depends(get_session),
) -> AsyncIterator[MemoryVectorSearch]:
    yield VectorSearch(session)


class _DbMemoryRecords:
    """기억 검색 결과의 상세 — 백엔드 MySQL(읽기 전용)에서 읽는다."""

    async def get_many(self, *, user_id: int, record_ids: list[int]) -> dict[int, MemoryRecord]:
        try:
            async with mysql.get_sessionmaker()() as mysql_session:
                return await MemoryRecordRepository(mysql_session).get_many(
                    user_id=user_id, record_ids=record_ids
                )
        except SQLAlchemyError as e:
            raise UpstreamError("백엔드 DB에서 자물쇠 상세를 읽지 못했어요.") from e

    async def list_condition_rows(self, *, user_id: int, limit: int) -> list[ConditionRow]:
        try:
            async with mysql.get_sessionmaker()() as mysql_session:
                return await MemoryRecordRepository(mysql_session).list_condition_rows(
                    user_id=user_id, limit=limit
                )
        except SQLAlchemyError as e:
            raise UpstreamError("백엔드 DB에서 자물쇠 정보를 읽지 못했어요.") from e


class _NoMemoryRecords:
    """백엔드 DB가 없는 로컬·테스트 환경 — 상세를 못 읽어서 결과가 비어 있다."""

    async def get_many(self, *, user_id: int, record_ids: list[int]) -> dict[int, MemoryRecord]:
        return {}

    async def list_condition_rows(self, *, user_id: int, limit: int) -> list[ConditionRow]:
        return []


def get_memory_record_source() -> MemoryRecordSource:
    s = get_settings()
    if s.ai_client_mode == "fake" or not s.backend_mysql_url:
        return _NoMemoryRecords()
    return _DbMemoryRecords()


@lru_cache
def get_question_parser() -> QuestionParser:
    # 사전·코드표·규칙 파일은 프로세스당 한 번만 읽는다
    return QuestionParser.load_default()


async def get_memory_search_service(
    search: MemoryVectorSearch = Depends(get_memory_vector_search),
    records: MemoryRecordSource = Depends(get_memory_record_source),
) -> MemorySearchService:
    s = get_settings()
    return MemorySearchService(
        get_embedder(),
        search,
        records,
        min_similarity=s.memory_min_similarity,
        comment_weight=s.memory_search_comment_weight,
        music_weight=s.memory_search_music_weight,
        candidates=s.memory_search_candidates,
        parser=get_question_parser(),
        condition_weight=s.memory_search_condition_weight,
        condition_weights=ConditionWeights(
            location=s.memory_search_location_weight,
            other=s.memory_search_other_condition_weight,
        ),
        scan_limit=s.memory_search_scan_limit,
        condition_candidates=s.memory_search_condition_candidates,
        display_count=s.memory_search_display_count,
        suggestion_count=s.memory_search_suggestion_count,
        # 답변 앞머리 문장 — 꺼 두면(기본) 템플릿 문장. LLM에는 조건 이름과 건수만 보낸다
        headline_writer=(
            HeadlineWriter(get_clients().llm_general) if s.memory_answer_llm_enabled else None
        ),
        headline_timeout=s.memory_answer_llm_timeout_seconds,
    )


@lru_cache
@lru_cache
def get_kure_model() -> TextEmbeddingModel | None:
    """임베딩 서비스(embedding_main)가 쓰는 KURE-v1 모델. 프로세스당 하나만 올린다.

    모델 파일이 없으면 None이라 서버는 뜨고 임베딩 요청만 503이 난다.
    """
    s = get_settings()
    if s.ai_client_mode == "fake":
        return FakeKureModel(dim=s.kure_embedding_dim, model=s.kure_model_name)
    return load_kure_model(
        s.kure_model_dir, s.kure_model_name, s.kure_embedding_dim, s.kure_threads, s.kure_max_tokens
    )


def get_moderation_model() -> ModerationModel | None:
    # 프로세스당 한 번만 로드한다 (ONNX 세션 생성에 1~2초)
    s = get_settings()
    if s.ai_client_mode == "fake":
        return FakeModerationModel()
    return load_onnx_model(s.moderation_model_dir, s.moderation_model_version, s.moderation_threads)


def get_moderation_service() -> ModerationService:
    return ModerationService(get_moderation_model(), get_settings().moderation_threshold)


class _DbGenreStats:
    """오늘의 질문 재료 — 백엔드 MySQL(자물쇠)과 AI Postgres(곡 장르·무드)를 함께 읽는다."""

    async def get(self, music_genre: str) -> GenreStats:
        s = get_settings()
        async with (
            mysql.get_sessionmaker()() as mysql_session,
            postgres.get_sessionmaker()() as pg_session,
        ):
            repo = BalanceStatsRepository(mysql_session, pg_session)
            return await repo.get(
                music_genre, days=s.balance_stats_days, limit=s.balance_stats_limit
            )


class _NoGenreStats:
    """백엔드 DB가 없는 로컬·테스트 환경 — 통계 없이 일반 취향 질문을 만든다."""

    async def get(self, music_genre: str) -> GenreStats:
        return GenreStats.empty()


def get_genre_stats_source() -> GenreStatsSource:
    s = get_settings()
    if s.ai_client_mode == "fake" or not s.backend_mysql_url:
        return _NoGenreStats()
    return _DbGenreStats()


def get_balance_service() -> BalanceService:
    # 곡·가수 지식이 필요한 작업이라 knowledge LLM을 쓴다
    return BalanceService(get_genre_stats_source(), QuestionGenerator(get_clients().llm_knowledge))
