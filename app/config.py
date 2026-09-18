"""환경변수·외부 API 키 설정."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"

    # 백엔드 → AI 서버 내부 인증 (X-Internal-Token)
    internal_token: str = "dev-internal-token"

    # AI 전용 PostgreSQL (pgvector)
    database_url: str = "postgresql+asyncpg://muro:muro@localhost:5432/muro_ai"

    # fake: 키 없이 결정적 가짜 벡터·텍스트로 동작 (로컬 개발·테스트용)
    # real: 실제 외부 API 호출
    ai_client_mode: Literal["fake", "real"] = "fake"

    # LLM (음악 무드 태깅 등)
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"

    # 텍스트 임베딩 (코멘트·음악 무드 텍스트)
    openai_api_key: str = ""
    text_embedding_model: str = "text-embedding-3-small"
    text_embedding_dim: int = 1536

    # CLIP 이미지 임베딩 — 제공처·모델·차원은 결정 회의 후 확정
    clip_api_url: str = ""
    clip_api_key: str = ""
    clip_model: str = "clip-vit-base-patch32"
    clip_embedding_dim: int = 512

    # 외부 API 공통
    external_timeout_seconds: float = 20.0
    external_max_retries: int = 2

    # 워커 (Postgres 기반 작업 큐)
    worker_poll_interval_seconds: float = 1.0
    job_max_attempts: int = 3
    job_stale_after_seconds: int = 600


@lru_cache
def get_settings() -> Settings:
    return Settings()
