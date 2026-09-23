"""AI PostgreSQL 테이블 정의."""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import BigInteger, DateTime, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.config import get_settings

_settings = get_settings()


class Base(DeclarativeBase):
    pass


class RecordEmbedding(Base):
    """자물쇠 1건 = 1행. 세 벡터는 좌표계가 다르다 (사진은 CLIP, 나머지는 텍스트 임베딩)."""

    __tablename__ = "record_embeddings"

    record_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    external_track_id: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    image_embedding: Mapped[list[float]] = mapped_column(Vector(_settings.clip_embedding_dim))
    music_embedding: Mapped[list[float]] = mapped_column(Vector(_settings.text_embedding_dim))
    comment_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(_settings.text_embedding_dim), nullable=True
    )
    music_mood_text: Mapped[str] = mapped_column(Text)
    model_versions: Mapped[dict[str, Any]] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TrackMood(Base):
    """곡 단위 무드 묘사 캐시. 같은 곡은 LLM을 한 번만 부른다.

    곡명·가수 원문은 저장하지 않고 스포티파이 곡 ID로만 식별한다.
    mood_version이 현재 설정(LLM·프롬프트·임베딩 모델)과 다르면 다시 만든다.
    """

    __tablename__ = "track_moods"

    external_track_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mood_text: Mapped[str] = mapped_column(Text)
    mood_embedding: Mapped[list[float]] = mapped_column(Vector(_settings.text_embedding_dim))
    mood_version: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class AiJob(Base):
    """게이트웨이와 워커 사이의 작업 큐. 상태·재시도 이력을 함께 남긴다."""

    __tablename__ = "ai_jobs"
    __table_args__ = (
        UniqueConstraint("job_type", "dedupe_key", name="uq_ai_jobs_type_key"),
        Index("ix_ai_jobs_status_run_after", "status", "run_after"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    job_type: Mapped[str] = mapped_column(String(32))
    dedupe_key: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # pending → running → done | failed (재시도 남으면 다시 pending) | cancelled (자물쇠 삭제)
    status: Mapped[str] = mapped_column(String(16), server_default="pending")
    attempts: Mapped[int] = mapped_column(Integer, server_default="0")
    max_attempts: Mapped[int] = mapped_column(Integer)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
