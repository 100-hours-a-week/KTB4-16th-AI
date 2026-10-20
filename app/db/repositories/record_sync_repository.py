"""기능4 자물쇠 동기화 — 백엔드 MySQL에서 바뀐 자물쇠를 읽고, 읽은 위치를 AI Postgres에 기억한다.

읽는 값은 백엔드가 원래 POST /api/embeddings/generate로 보내 주기로 했던 것
(recordId, userId, 사진, 곡, 코멘트, 작성 시각) + 변경·삭제를 알아채는 updated_at·deleted_at
+ 요청 본문에는 없던 값(날씨·기온·기분·장소). 뒤의 값들은 기억 검색이 읽는 record_facts에 쓴다.
"""

from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time
from decimal import Decimal
from typing import Any

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db.models import SyncState

SYNC_NAME = "records"
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Cursor:
    updated_at: datetime
    record_id: int


@dataclass(frozen=True)
class ChangedRecord:
    record_id: int
    user_id: int
    photo_key: str | None
    title: str
    artist_name: str
    external_track_id: str
    comment: str | None
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None
    # 요청 본문에 없던 값 — 기억 검색(record_facts)용. 비어 있을 수 있어서 모두 기본값 None
    recorded_at_local: datetime | None = None  # records.created_at 그대로 (한국시간, 시간대 없음)
    weather_condition: str | None = None
    temperature: float | None = None
    mood_score: int | None = None
    dong_name: str | None = None
    dong_code: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    raw_record: dict[str, Any] | None = None  # records 한 행의 모든 칸 (JSON으로 바꾼 원문)


def jsonable(value: Any) -> Any:
    """DB 값을 JSON에 담을 수 있게 바꾼다 (시각은 글자로, Decimal은 숫자로, bytes는 16진수로)."""
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def _to_float(value: object) -> float | None:
    # MySQL DECIMAL은 Decimal로 온다
    return None if value is None else float(value) if isinstance(value, Decimal) else value  # type: ignore[return-value]


def _aware(value: datetime) -> datetime:
    # MySQL DATETIME은 시간대 정보가 없다 — 백엔드 DB 시각은 UTC로 저장된다고 보고 붙인다
    return value if value.tzinfo else value.replace(tzinfo=UTC)


class BackendRecordReader:
    def __init__(self, mysql_session: AsyncSession):
        self._mysql = mysql_session

    async def changed_since(self, cursor: Cursor, limit: int) -> list[ChangedRecord]:
        """커서 다음부터 바뀐 자물쇠를 (바뀐 시각, record_id) 순으로. 삭제된 것도 포함.

        바뀐 시각 = updated_at과 deleted_at 중 늦은 쪽 — 소프트 삭제 때 updated_at을 안
        건드리는 경우에도 삭제를 놓치지 않게.
        """
        s = get_settings()
        sql = f"""
            SELECT * FROM (
                SELECT r.record_id, r.user_id, r.comment, r.created_at, r.deleted_at,
                       r.weather_condition, r.temperature, r.mood_score,
                       GREATEST(r.updated_at, COALESCE(r.deleted_at, r.updated_at)) AS changed_at,
                       m.title, m.artist_name, m.external_track_id,
                       p.legal_dong_name, p.legal_dong_code, p.latitude, p.longitude,
                       ph.{s.backend_photo_key_column} AS photo_key
                FROM records r
                JOIN music_tracks m ON r.music_track_id = m.music_track_id
                LEFT JOIN places p ON r.place_id = p.place_id
                LEFT JOIN {s.backend_photo_table} ph
                       ON ph.{s.backend_photo_record_column} = r.record_id
            ) t
            WHERE t.changed_at > :ts OR (t.changed_at = :ts AND t.record_id > :rid)
            ORDER BY t.changed_at, t.record_id
            LIMIT :limit
        """  # noqa: S608 — 테이블·칸 이름은 설정값(사용자 입력 아님)
        ts = cursor.updated_at.astimezone(UTC).replace(tzinfo=None)
        rows = await self._mysql.execute(
            text(sql), {"ts": ts, "rid": cursor.record_id, "limit": limit}
        )
        records = [
            ChangedRecord(
                record_id=row.record_id,
                user_id=row.user_id,
                photo_key=row.photo_key,
                title=row.title,
                artist_name=row.artist_name,
                external_track_id=row.external_track_id,
                comment=row.comment,
                created_at=_aware(row.created_at),
                updated_at=_aware(row.changed_at),
                deleted_at=_aware(row.deleted_at) if row.deleted_at else None,
                recorded_at_local=row.created_at,
                weather_condition=row.weather_condition,
                temperature=_to_float(row.temperature),
                mood_score=None if row.mood_score is None else int(row.mood_score),
                dong_name=row.legal_dong_name,
                dong_code=None if row.legal_dong_code is None else str(row.legal_dong_code),
                latitude=_to_float(row.latitude),
                longitude=_to_float(row.longitude),
            )
            for row in rows
        ]
        if records and s.record_facts_store_raw:
            raw = await self._raw_rows([r.record_id for r in records])
            records = [replace(r, raw_record=raw.get(r.record_id)) for r in records]
        return records

    async def _raw_rows(self, record_ids: list[int]) -> dict[int, dict[str, Any]]:
        """records 한 행의 모든 칸 — 칸 이름을 하나씩 적지 않고 통째로 읽어 JSON으로 바꾼다."""
        rows = await self._mysql.execute(
            text("SELECT * FROM records WHERE record_id IN :ids").bindparams(
                bindparam("ids", expanding=True)
            ),
            {"ids": record_ids},
        )
        return {row.record_id: jsonable(dict(row._mapping)) for row in rows}


class SyncCursorStore:
    def __init__(self, postgres_session: AsyncSession):
        self._pg = postgres_session

    async def load(self) -> Cursor:
        row = await self._pg.get(SyncState, SYNC_NAME)
        if row is None:
            return Cursor(EPOCH, 0)  # 처음이면 전체 백필
        return Cursor(row.last_updated_at, row.last_record_id)

    async def save(self, cursor: Cursor) -> None:
        stmt = insert(SyncState).values(
            name=SYNC_NAME, last_updated_at=cursor.updated_at, last_record_id=cursor.record_id
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["name"],
            set_={
                "last_updated_at": stmt.excluded.last_updated_at,
                "last_record_id": stmt.excluded.last_record_id,
                "updated_at": text("now()"),
            },
        )
        await self._pg.execute(stmt)
        await self._pg.commit()
