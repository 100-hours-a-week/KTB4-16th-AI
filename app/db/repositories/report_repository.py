"""RECAP 생성에 필요한 데이터 조회 — 백엔드 MySQL(자물쇠 원본) + AI Postgres(무드 텍스트)."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import MonthlyReport, RecordEmbedding


@dataclass(frozen=True)
class RecordSummary:
    """자물쇠 1건 — RECAP 생성에 필요한 필드. (백엔드 records/music/place 테이블)

    자물쇠 생성 시 저장되는 값 중 요약에 쓸 수 있는 것들을 그대로 가져온다.
    music_mood_text·사진 카테고리는 AI 자체 DB(record_embeddings)에서 별도 조회한다.

    테이블·컬럼명은 백엔드 Flyway 마이그레이션(V1~V4) 기준: records, music_tracks, places.
    places엔 장소 이름 컬럼이 없고(V4에서 place_name 삭제) 법정동 이름(legal_dong_name)만
    있어서 place_name은 그 값이다. 비어 있을 수 있다.
    """

    record_id: int
    external_track_id: str
    created_at: datetime
    mood_score: int | None  # 사용자가 직접 입력한 기분 점수 (-50~50)
    weather_condition: str | None
    temperature: float | None  # DECIMAL(3,1) — 18.5도처럼 소수가 온다
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
        자물쇠 삭제는 소프트 삭제(deleted_at)라 삭제된 기록은 뺀다.
        """
        rows = await self._mysql.execute(
            text(
                """
                SELECT DISTINCT user_id
                FROM records
                WHERE deleted_at IS NULL
                  AND YEAR(created_at) = :year AND MONTH(created_at) = :month
                """
            ),
            {"year": year, "month": month},
        )
        return [row.user_id for row in rows]

    async def get_records(self, *, user_id: int, year: int, month: int) -> list[RecordSummary]:
        """특정 사용자의 해당 달 자물쇠 목록. (백엔드 MySQL)

        records.music_track_id·place_id는 NOT NULL FK라 JOIN으로 곡·장소를 붙인다.
        """
        rows = await self._mysql.execute(
            text(
                """
                SELECT r.record_id, m.external_track_id, r.created_at,
                       r.mood_score, r.weather_condition, r.temperature, r.comment,
                       m.artist_name, p.legal_dong_name AS place_name
                FROM records r
                JOIN music_tracks m ON r.music_track_id = m.music_track_id
                JOIN places p ON r.place_id = p.place_id
                WHERE r.user_id = :user_id AND r.deleted_at IS NULL
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
                temperature=float(row.temperature) if row.temperature is not None else None,
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
        """
        avg_row = await self._mysql.execute(
            text(
                """
                SELECT AVG(mood_score) AS avg_mood
                FROM records
                WHERE user_id = :user_id AND deleted_at IS NULL
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
                    JOIN music_tracks m ON r.music_track_id = m.music_track_id
                    WHERE r.user_id = :user_id AND r.deleted_at IS NULL
                      AND YEAR(r.created_at) = :year AND MONTH(r.created_at) = :month
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
                    SELECT p.legal_dong_name AS name, COUNT(*) AS cnt,
                           MAX(r.created_at) AS latest
                    FROM records r
                    JOIN places p ON r.place_id = p.place_id
                    WHERE r.user_id = :user_id AND r.deleted_at IS NULL
                      AND YEAR(r.created_at) = :year AND MONTH(r.created_at) = :month
                      AND p.legal_dong_name IS NOT NULL
                    GROUP BY p.legal_dong_name
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


@dataclass(frozen=True)
class MonthlyReportRecord:
    """저장된 RECAP 스냅샷 1건. 조회 API 응답을 만드는 데 쓴다.

    기분·아티스트·장소 통계는 안 담는다 — 백엔드가 자기 MySQL 원본으로 직접 계산하기로
    협의됨(AI는 요약 텍스트를 쓸 때만 내부적으로 계산해 쓰고 저장은 안 함).
    """

    ai_recap_text: str
    photo_scenes: list[dict[str, Any]]


class MonthlyReportRepository:
    """RECAP 결과 저장·조회. AI 자체 Postgres(monthly_reports)에만 접근한다.

    백엔드 MySQL은 안 건드린다 — 배치 생성(ReportRepository)과는 별개 관심사라 분리했다.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(
        self,
        *,
        user_id: int,
        year: int,
        month: int,
        ai_recap_text: str,
        photo_scenes: list[dict[str, Any]],
    ) -> None:
        """이미 있으면 덮어쓴다 — job 재시도로 같은 달이 다시 들어올 수 있다."""
        stmt = insert(MonthlyReport).values(
            user_id=user_id,
            year=year,
            month=month,
            ai_recap_text=ai_recap_text,
            photo_scenes=photo_scenes,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["user_id", "year", "month"],
            set_={
                "ai_recap_text": stmt.excluded.ai_recap_text,
                "photo_scenes": stmt.excluded.photo_scenes,
            },
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def get(self, *, user_id: int, year: int, month: int) -> MonthlyReportRecord | None:
        row = (
            await self._session.execute(
                select(MonthlyReport).where(
                    MonthlyReport.user_id == user_id,
                    MonthlyReport.year == year,
                    MonthlyReport.month == month,
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return MonthlyReportRecord(
            ai_recap_text=row.ai_recap_text,
            photo_scenes=list(row.photo_scenes),
        )

    async def get_many(
        self, *, user_ids: list[int], year: int, month: int
    ) -> dict[int, MonthlyReportRecord]:
        """배치 완료 콜백에 여러 유저 결과를 한 번에 실어 보낼 때 쓴다."""
        if not user_ids:
            return {}
        rows = await self._session.execute(
            select(MonthlyReport).where(
                MonthlyReport.user_id.in_(user_ids),
                MonthlyReport.year == year,
                MonthlyReport.month == month,
            )
        )
        return {
            row.user_id: MonthlyReportRecord(
                ai_recap_text=row.ai_recap_text, photo_scenes=list(row.photo_scenes)
            )
            for row in rows.scalars()
        }
