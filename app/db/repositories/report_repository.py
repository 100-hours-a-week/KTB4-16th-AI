"""RECAP 생성에 필요한 데이터 조회 — 백엔드 MySQL(자물쇠 원본) + AI Postgres(무드 텍스트)."""

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RecordEmbedding


@dataclass(frozen=True)
class RecordSummary:
    """자물쇠 1건 — RECAP 생성에 필요한 필드. (백엔드 records/music/place 테이블)

    자물쇠 생성 시 저장되는 값 중 요약에 쓸 수 있는 것들을 그대로 가져온다.
    music_mood_text·사진 카테고리는 AI 자체 DB(record_embeddings)에서 별도 조회한다.

    TODO: 위키 모델 API 설계 확정 후 실제 컬럼에 맞춰 필드 조정.
    artist_name·place_name은 records 자체가 아니라 music/place 테이블에 있다고
    추정하고 JOIN으로 가져옴 — 실제 테이블/FK명은 백엔드 확인 필요.
    """

    record_id: int
    external_track_id: str
    created_at: datetime
    mood_score: int | None  # 사용자가 직접 입력한 기분 점수 (-50~50)
    weather_condition: str | None
    temperature: int | None
    comment: str | None
    artist_name: str | None
    place_name: str | None


@dataclass(frozen=True)
class ReportStats:
    """한 사용자의 한 달치 집계 통계 — RecordSummary(자물쇠 1건당 1개)와 달리 이건 1개만 존재."""

    avg_mood_score: float | None
    top_artist: str | None
    top_place: str | None


class ReportRepository:
    def __init__(self, mysql_session: AsyncSession, postgres_session: AsyncSession):
        self._mysql = mysql_session
        self._postgres = postgres_session

    async def get_active_user_ids(self, *, year: int, month: int) -> list[int]:
        """이번 달(year, month)에 자물쇠 기록이 있는 전체 사용자 ID. (백엔드 MySQL)

        user_ids 생략 요청 시 배치 대상 전체를 정하는 데 쓴다.
        TODO: 실제 테이블/컬럼명은 백엔드 records 스키마 확정 후 채운다.
        """
        rows = await self._mysql.execute(
            text(
                """
                SELECT DISTINCT user_id
                FROM records
                WHERE YEAR(created_at) = :year AND MONTH(created_at) = :month
                """
            ),
            {"year": year, "month": month},
        )
        return [row.user_id for row in rows]

    async def get_records(self, *, user_id: int, year: int, month: int) -> list[RecordSummary]:
        """특정 사용자의 해당 달 자물쇠 목록. (백엔드 MySQL)

        TODO: 실제 테이블/컬럼명은 백엔드 records 스키마 확정 후 채운다.
        music/place는 records에 직접 컬럼이 없고 FK로 연결된다고 추정(records.music_track_id
        -> music.id, records.place_id -> place.id). LEFT JOIN이라 추정이 틀려도 records
        자체는 그대로 나오고 artist_name/place_name만 NULL로 빠진다.
        """
        rows = await self._mysql.execute(
            text(
                """
                SELECT r.id AS record_id, m.external_track_id, r.created_at,
                       r.mood_score, r.weather_condition, r.temperature, r.comment,
                       m.artist_name, p.name AS place_name
                FROM records r
                LEFT JOIN music m ON r.music_track_id = m.id
                LEFT JOIN place p ON r.place_id = p.id
                WHERE r.user_id = :user_id
                  AND YEAR(r.created_at) = :year AND MONTH(r.created_at) = :month
                ORDER BY r.created_at
                """
            ),
            {"user_id": user_id, "year": year, "month": month},
        )
        return [
            RecordSummary(
                record_id=row.record_id,
                external_track_id=row.external_track_id,
                created_at=row.created_at,
                mood_score=row.mood_score,
                weather_condition=row.weather_condition,
                temperature=row.temperature,
                comment=row.comment,
                artist_name=row.artist_name,
                place_name=row.place_name,
            )
            for row in rows
        ]

    async def get_stats(self, *, user_id: int, year: int, month: int) -> ReportStats:
        """이 사용자의 이번 달 평균 기분·TOP 아티스트·TOP 장소. (백엔드 MySQL)

        TOP 아티스트/장소는 기록 수 DESC, 동률이면 최근 기록 시각 DESC로 정렬한다
        (MULO API 스펙의 "장소별 인기 음악 조회"가 쓰는 동률 처리 규칙과 동일하게 맞춤).
        TODO: get_records와 동일한 music/place 스키마 추정.
        """
        avg_row = await self._mysql.execute(
            text(
                """
                SELECT AVG(mood_score) AS avg_mood
                FROM records
                WHERE user_id = :user_id
                  AND YEAR(created_at) = :year AND MONTH(created_at) = :month
                """
            ),
            {"user_id": user_id, "year": year, "month": month},
        )
        avg_mood = avg_row.scalar_one_or_none()

        artist_row = (
            await self._mysql.execute(
                text(
                    """
                    SELECT m.artist_name AS name, COUNT(*) AS cnt, MAX(r.created_at) AS latest
                    FROM records r
                    LEFT JOIN music m ON r.music_track_id = m.id
                    WHERE r.user_id = :user_id
                      AND YEAR(r.created_at) = :year AND MONTH(r.created_at) = :month
                      AND m.artist_name IS NOT NULL
                    GROUP BY m.artist_name
                    ORDER BY cnt DESC, latest DESC
                    LIMIT 1
                    """
                ),
                {"user_id": user_id, "year": year, "month": month},
            )
        ).first()

        place_row = (
            await self._mysql.execute(
                text(
                    """
                    SELECT p.name AS name, COUNT(*) AS cnt, MAX(r.created_at) AS latest
                    FROM records r
                    LEFT JOIN place p ON r.place_id = p.id
                    WHERE r.user_id = :user_id
                      AND YEAR(r.created_at) = :year AND MONTH(r.created_at) = :month
                      AND p.name IS NOT NULL
                    GROUP BY p.name
                    ORDER BY cnt DESC, latest DESC
                    LIMIT 1
                    """
                ),
                {"user_id": user_id, "year": year, "month": month},
            )
        ).first()

        return ReportStats(
            avg_mood_score=float(avg_mood) if avg_mood is not None else None,
            top_artist=artist_row.name if artist_row else None,
            top_place=place_row.name if place_row else None,
        )

    async def get_mood_texts(self, record_ids: list[int]) -> dict[int, str]:
        """자물쇠 ID들의 music_mood_text. (AI Postgres record_embeddings)

        아직 임베딩이 생성되지 않은 record_id는 결과에서 빠진다 — 호출부에서 누락 처리.
        """
        if not record_ids:
            return {}
        rows = await self._postgres.execute(
            select(RecordEmbedding.record_id, RecordEmbedding.music_mood_text).where(
                RecordEmbedding.record_id.in_(record_ids)
            )
        )
        return {row.record_id: row.music_mood_text for row in rows}

    async def get_image_embeddings(self, record_ids: list[int]) -> dict[int, list[float]]:
        """자물쇠 ID들의 image_embedding. (AI Postgres record_embeddings)

        기능4가 자물쇠 생성 시점에 이미 CLIP으로 만들어둔 사진 벡터를 그대로 재사용한다
        — RECAP 사진 분류를 위해 CLIP을 다시 호출하지 않는다.
        아직 임베딩이 생성되지 않은 record_id는 결과에서 빠진다 — 호출부에서 누락 처리.
        """
        if not record_ids:
            return {}
        rows = await self._postgres.execute(
            select(RecordEmbedding.record_id, RecordEmbedding.image_embedding).where(
                RecordEmbedding.record_id.in_(record_ids)
            )
        )
        return {row.record_id: list(row.image_embedding) for row in rows}