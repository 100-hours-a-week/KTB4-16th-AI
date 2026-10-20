"""AI PostgreSQL 테이블 정의."""

from datetime import date, datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    SmallInteger,
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
    # 텍스트 임베딩 칸은 모델을 바꾸면(0010: OpenAI → KURE) 차원이 달라져 기존 값을 비운다.
    # 재임베딩(app/tools/reembed_text.py)이 끝나기 전에는 NULL이라 NOT NULL을 풀었다.
    music_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(_settings.text_vector_dim), nullable=True
    )
    comment_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(_settings.text_vector_dim), nullable=True
    )
    music_mood_text: Mapped[str] = mapped_column(Text)
    model_versions: Mapped[dict[str, Any]] = mapped_column(JSONB)
    # 사진 분류 결과 (축 이름 → 태그). NULL은 분류 안 함/실패, {}는 붙일 태그 없음.
    # none_as_null: 파이썬 None을 JSON null이 아니라 SQL NULL로 저장한다
    photo_tags: Mapped[dict[str, str] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class RecordFact(Base):
    """자물쇠 1건의 조건 검색용 값 — 백엔드 MySQL(records·places·music_tracks)에서 옮겨 온 사본.

    기억 검색이 날짜·날씨·기온·기분·장소·곡명·가수를 질문마다 백엔드에서 읽지 않고 여기서 읽는다
    (설정 MEMORY_RECORD_SOURCE, 기본 postgres). 원본은 백엔드 DB이고, 이 표는 자물쇠 동기화
    (record_sync_worker)가 맞춘다.
    recorded_at은 백엔드 created_at 그대로(한국시간)라 시간대 없는 칸이다.
    comment_text는 설정 RECORD_FACTS_STORE_COMMENT(기본 저장)로 정한다(개인정보). 사전에 없는
    장소를 코멘트 글자로 찾을 때 쓴다.
    """

    __tablename__ = "record_facts"
    __table_args__ = (Index("ix_record_facts_user_recorded", "user_id", "recorded_at"),)

    record_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    user_id: Mapped[int] = mapped_column(BigInteger)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=False))
    weather_condition: Mapped[str | None] = mapped_column(String(30), nullable=True)
    temperature: Mapped[float | None] = mapped_column(Float(53), nullable=True)
    mood_score: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)
    dong_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    dong_code: Mapped[str | None] = mapped_column(String(10), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Float(53), nullable=True)
    longitude: Mapped[float | None] = mapped_column(Float(53), nullable=True)
    comment_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    track_title: Mapped[str | None] = mapped_column(String(300), nullable=True)
    artist_name: Mapped[str | None] = mapped_column(String(300), nullable=True)
    # 백엔드 records 한 행의 모든 칸을 원문 그대로 (칸이 늘어도 받는다). NULL은 아직 안 채운 행
    raw_record: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class TrackMood(Base):
    """곡 단위 무드 묘사 캐시. 같은 곡은 LLM을 한 번만 부른다.

    곡명·가수 원문은 저장하지 않고 스포티파이 곡 ID로만 식별한다.
    mood_version이 현재 설정(LLM·프롬프트·임베딩 모델)과 다르면 다시 만든다.
    """

    __tablename__ = "track_moods"

    external_track_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    mood_text: Mapped[str] = mapped_column(Text)
    # 0010에서 NOT NULL을 풀었다 (재임베딩 전에는 NULL). NULL인 행은 캐시로 쓰지 않는다
    mood_embedding: Mapped[list[float] | None] = mapped_column(
        Vector(_settings.text_vector_dim), nullable=True
    )
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

    찾은 곡은 기한 없이 쓰고, 없는 곡만 verified_at 기준으로 하루가 지나면 다시 확인한다
    (track_resolver의 TTL 참고).
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


class UserMusicGenre(Base):
    """기능6 사용자별 취향 투표 그룹 — 매일 배치가 계산, 백엔드는 뷰(be_user_music_genres)로 읽는다.

    값은 13개 장르 + '공통'. 같은 그룹 사용자끼리 같은 질문을 받고 그 안에서 매칭된다.
    """

    __tablename__ = "user_music_genres"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=False)
    music_genre: Mapped[str] = mapped_column(String(50))
    # records: 저장한 곡 장르 1등 / common: 장르 정보가 없거나 그룹 인원이 적어 공통
    # merged: 공통 그룹도 작아서 가장 큰 장르 그룹에 합쳐짐
    source: Mapped[str] = mapped_column(String(20))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class VoteQuestion(Base):
    """기능6 그룹별 오늘의 질문 — 백엔드는 approved만 뷰(be_vote_questions)로 읽는다.

    (question_date, music_genre)마다 approved 질문은 1개 (부분 유니크 인덱스).
    사람이 검수해서 rejected로 바꾸면 다음 배치가 그 자리를 다시 채운다.
    """

    __tablename__ = "vote_questions"
    __table_args__ = (Index("ix_vote_questions_date_genre", "question_date", "music_genre"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question_date: Mapped[date] = mapped_column(Date)
    music_genre: Mapped[str] = mapped_column(String(50))
    question: Mapped[str] = mapped_column(String(100))
    option_a: Mapped[str] = mapped_column(String(50))
    option_b: Mapped[str] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(16), server_default="approved")
    model_version: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class SyncState(Base):
    """백엔드 DB에서 어디까지 읽었는지 기억하는 표 (기능4 자물쇠 동기화).

    (updated_at, record_id) 쌍으로 기억해서, 같은 시각에 바뀐 자물쇠가 여러 개여도
    빠뜨리거나 두 번 처리하지 않는다.
    """

    __tablename__ = "sync_state"

    name: Mapped[str] = mapped_column(String(50), primary_key=True)
    last_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_record_id: Mapped[int] = mapped_column(BigInteger, server_default="0")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ArtistAlias(Base):
    """가수 이름의 다른 표기(영어↔한글) 목록. 기억 검색이 질문에서 가수를 찾을 때 함께 비교한다.

    가수 이름은 공백·기호·대소문자를 뺀 글자(name_key)로 구분한다. aliases가 빈 목록이면
    "LLM이 모른다고 답함"이라는 결과라서 같은 alias_version에서는 다시 묻지 않는다.
    alias_version이 현재 설정(LLM·프롬프트)과 다르면 다시 만든다. 'manual'은 사람이 직접 넣은
    행이라 다시 만들지 않는다.
    """

    __tablename__ = "artist_aliases"

    name_key: Mapped[str] = mapped_column(String(200), primary_key=True)
    artist_name: Mapped[str] = mapped_column(String(200))
    aliases: Mapped[list[str]] = mapped_column(ARRAY(String(100)), server_default="{}")
    alias_version: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
