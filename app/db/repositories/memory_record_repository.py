"""기능5 기억 검색 — 찾은 자물쇠의 상세를 백엔드 MySQL(읽기 전용)에서 읽는다.

AI DB에는 벡터와 곡 ID만 있고 날짜·장소·곡명은 백엔드 DB에 있다. 리포트(RECAP)가 이미 쓰는
읽기 전용 연결로 읽으므로 백엔드 코드는 안 바뀐다.

사용자 격리: SQL에 user_id 조건을 넣어서 다른 사람의 자물쇠는 id를 알아도 읽히지 않는다.
삭제된 자물쇠(deleted_at)는 뺀다 — AI DB 벡터는 백엔드가 삭제 호출을 보낼 때까지 남을 수 있다.

테이블·컬럼명은 백엔드 Flyway 마이그레이션 기준이다. records·places의 컬럼은 리포트 코드로 확인했고,
music_tracks의 title·album_image_url·external_url은 응답 스키마 주석(백엔드 컬럼과 같게 맞춤)을
근거로 한 것이라 실제 DB로 한 번 확인이 필요하다.

조건 검색(list_condition_rows)이 읽는 places.latitude·longitude·legal_dong_code도 실제 DB로 확인이
필요하다(API 응답 필드 이름을 따라 추정한 것). records.weather_condition·temperature·mood_score는
리포트 코드가 이미 읽는 이름이다. 추정이 틀려도 get_many는 이 컬럼을 읽지 않아서, 조건 검색이
아닌 질문의 검색은 영향을 받지 않는다.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class MemoryRecord:
    record_id: int
    created_at: datetime
    title: str
    artist_name: str
    external_track_id: str
    album_image_url: str
    external_url: str
    place_id: int | None
    place_name: str | None  # 백엔드 places.legal_dong_name — 비어 있을 수 있다


@dataclass(frozen=True)
class ConditionRow:
    """조건 점수 계산에 필요한 값만 담은 가벼운 자물쇠 정보. 비어 있을 수 있는 값은 None이다."""

    record_id: int
    created_at: datetime
    weather_condition: str | None
    temperature: float | None
    mood_score: int | None
    dong_name: str | None  # places.legal_dong_name
    dong_code: str | None  # places.legal_dong_code (10자리)
    latitude: float | None
    longitude: float | None
    # 글자 검색에 쓰는 글: 사용자가 쓴 코멘트(최대 80자). 곡 제목·가수 이름은 일부러 읽지 않는다
    comment: str | None = None


class MemoryRecordSource(Protocol):
    async def get_many(self, *, user_id: int, record_ids: list[int]) -> dict[int, MemoryRecord]: ...

    async def list_condition_rows(self, *, user_id: int, limit: int) -> list[ConditionRow]: ...


class MemoryRecordRepository:
    def __init__(self, mysql_session: AsyncSession):
        self._mysql = mysql_session

    async def get_many(self, *, user_id: int, record_ids: list[int]) -> dict[int, MemoryRecord]:
        if not record_ids:
            return {}
        rows = await self._mysql.execute(
            text(
                """
                SELECT r.record_id, r.created_at,
                       m.title, m.artist_name, m.external_track_id,
                       m.album_image_url, m.external_url,
                       p.place_id, p.legal_dong_name AS place_name
                FROM records r
                JOIN music_tracks m ON r.music_track_id = m.music_track_id
                JOIN places p ON r.place_id = p.place_id
                WHERE r.user_id = :user_id AND r.deleted_at IS NULL
                  AND r.record_id IN :record_ids
                """
            ).bindparams(bindparam("record_ids", expanding=True)),
            {"user_id": user_id, "record_ids": record_ids},
        )
        return {
            row.record_id: MemoryRecord(
                record_id=row.record_id,
                created_at=row.created_at,
                title=row.title,
                artist_name=row.artist_name,
                external_track_id=row.external_track_id,
                album_image_url=row.album_image_url or "",
                external_url=row.external_url or "",
                place_id=row.place_id,
                place_name=row.place_name,
            )
            for row in rows
        }

    async def list_condition_rows(self, *, user_id: int, limit: int) -> list[ConditionRow]:
        """이 사용자의 자물쇠를 최근순으로 limit개까지, 조건·코멘트 검색에 필요한 값만 읽는다.

        개인 일기라 자물쇠가 수천 개를 넘기 어렵다고 보고 SQL로 거르지 않고 가져와서 파이썬에서
        비교한다 — 조건 조합마다 SQL을 따로 만들 필요가 없고, 조건에 안 맞는 기록이 점수만
        낮아질 뿐 사라지지 않는다.
        """
        rows = await self._mysql.execute(
            text(
                """
                SELECT r.record_id, r.created_at, r.weather_condition, r.temperature,
                       r.mood_score, r.comment, p.legal_dong_name, p.legal_dong_code,
                       p.latitude, p.longitude
                FROM records r
                JOIN places p ON r.place_id = p.place_id
                WHERE r.user_id = :user_id AND r.deleted_at IS NULL
                ORDER BY r.created_at DESC, r.record_id DESC
                LIMIT :limit
                """
            ),
            {"user_id": user_id, "limit": limit},
        )
        return [
            ConditionRow(
                record_id=row.record_id,
                created_at=row.created_at,
                weather_condition=row.weather_condition,
                temperature=_to_float(row.temperature),
                mood_score=None if row.mood_score is None else int(row.mood_score),
                dong_name=row.legal_dong_name,
                dong_code=None if row.legal_dong_code is None else str(row.legal_dong_code),
                latitude=_to_float(row.latitude),
                longitude=_to_float(row.longitude),
                comment=row.comment,
            )
            for row in rows
        ]


def _to_float(value: object) -> float | None:
    # MySQL DECIMAL은 Decimal로 온다
    return None if value is None else float(value)  # type: ignore[arg-type]
