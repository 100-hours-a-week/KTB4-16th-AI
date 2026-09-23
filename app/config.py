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

    # 모델 서빙 계획 (README 참고)
    #   V1: 전부 외부 API — Anthropic(LLM), OpenAI(텍스트 임베딩), Replicate(CLIP)
    #   V2: GCP CPU(ONNX) — KcELECTRA, 텍스트 임베딩 / RunPod — CLIP, 로컬 LLM(general 작업만)

    # LLM — 작업 성격에 따라 두 갈래로 나눈다
    #   knowledge: 곡·가수 지식이 필요한 작업 (음악 무드 묘사, 투표 문항 생성) → 프론티어 유지
    #   general:   말 바꾸기만 하는 작업 (검색어 리라이팅, 요약, 답변, 클러스터 이름) → V2에 로컬
    llm_knowledge_provider: Literal["anthropic", "local"] = "anthropic"
    llm_general_provider: Literal["anthropic", "local"] = "anthropic"

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-5"

    # 로컬 LLM (V2, RunPod) — 구현 예정
    local_llm_url: str = ""
    local_llm_model: str = ""

    # 텍스트 임베딩 (코멘트·음악 무드 텍스트) — V2에 로컬 모델로 교체 예정
    openai_api_key: str = ""
    text_embedding_model: str = "text-embedding-3-small"
    text_embedding_dim: int = 1536

    # CLIP 이미지 임베딩 — Replicate openai/clip (clip-vit-large-patch14)
    # V2에 RunPod 직접 서빙으로 교체 예정. 모델이 바뀌면 차원 변경 → 전체 재임베딩
    replicate_api_token: str = ""
    clip_model: str = "openai/clip"
    clip_embedding_dim: int = 768

    # Spotify (앱 인증: 검색·곡·아티스트 조회)
    spotify_client_id: str = ""
    spotify_client_secret: str = ""

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
