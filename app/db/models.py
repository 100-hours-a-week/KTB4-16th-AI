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


class TrackLookup(Base):
    """LLM이 말한 곡(가수+제목) → Spotify 확인 결과 캐시.

    추천 1건마다 곡 8~10개를 Spotify에 조회하다가 하루 요청 한도를 넘겨 추천이 전부
    멈춘 적이 있다(2026-10-02, 429 QUOTA_EXCEEDED). 같은 곡은 자주 다시 나오므로 한 번
    확인한 결과를 저장해 두고 재사용한다.

    Spotify 개발자 약관은 메타데이터·커버의 "임시" 캐시만 허용하고 무기한 저장을 금지해서,
    verified_at 기준으로 유효기간이 지나면 다시 확인한다(track_resolver의 TTL 참고).
    external_track_id가 NULL이면 "Spotify에 없는 곡"(LLM이 지어낸 곡 등)이라는 기록이다.
    """

    __tablename__ = "track_lookups"

    lookup_key: Mapped[str] = mapped_column(String(300), primary_key=True)
    llm_artist: Mapped[str] = mapped_column(String(200))
    llm_title: Mapped[str] = mapped_column(String(300))
    genre: Mapped[str] = mapped_column(String(20), server_default="")
    moods: Mapped[list[str]] = mapped_column(ARRAY(String(20)), server_default="{}")
    external_track_id: Mapped[str | None] = mapped_column(String(64), index=True)
    title: Mapped[str | None] = mapped_column(String(300))
    artist_name: Mapped[str | None] = mapped_column(String(300))
    spotify_uri: Mapped[str | None] = mapped_column(String(100))
    album_image_url: Mapped[str | None] = mapped_column(Text)
    external_url: Mapped[str | None] = mapped_column(Text)
    verified_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


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

    배치가 끝나면 이 내용을 백엔드 콜백(MonthlyReportAiCallbackRequest)에 실어서
    바로 전달한다 — 별도 조회 API 없이 콜백 하나로 백엔드가 필요한 내용을 다 받는다.
    이 테이블은 콜백 전송 시점까지의 임시 보관 + 이후 재확인용 스냅샷 역할이다.

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
    # 백엔드 photoScenes — [{"tag": "카페", "count": 6, "ratio": 43}, ...] 형태, 비중 높은 순
    photo_scenes: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, server_default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
