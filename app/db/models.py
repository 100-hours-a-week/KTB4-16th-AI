"""AI PostgreSQL 테이블 정의."""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
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
    # 2026-09-26: 기능3 재랭킹이 임베딩 유사도 대신 이 태그 겹침으로 바뀌면서 추가.
    genre: Mapped[str] = mapped_column(String(20), server_default="")
    moods: Mapped[list[str]] = mapped_column(ARRAY(String(20)), server_default="{}")
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


class MonthlyReport(Base):
    """기능2 RECAP 결과 스냅샷. 월 종료 후 배치가 사용자당 1행만 만들고 이후 안 바뀐다.

    백엔드가 배치 완료 콜백을 받은 뒤 이 내용을 조회 API로 가져가서 자기 DB에 저장·서빙한다
    (AI API 시트 "월간 리포트(RECAP) 생성" 협의: 콜백은 완료 알림만, 내용은 AI 조회 API로 전달).

    기분·아티스트·장소 통계는 여기 안 둔다 — 백엔드가 자기 MySQL 원본으로 직접 계산하기로
    협의됨. AI는 요약 텍스트를 쓸 때만 내부적으로 그 값들을 계산해 쓰고 저장은 안 한다.
    """

    __tablename__ = "monthly_reports"
    __table_args__ = (
        UniqueConstraint("user_id", "year", "month", name="uq_monthly_reports_user_year_month"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    year: Mapped[int] = mapped_column(Integer)
    month: Mapped[int] = mapped_column(Integer)
    ai_recap_text: Mapped[str] = mapped_column(Text)
    # 백엔드 photoScenes — 이번 달 사진에서 나온 장면 카테고리, 빈도 높은 순
    photo_scenes: Mapped[list[str]] = mapped_column(ARRAY(String(20)), server_default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
