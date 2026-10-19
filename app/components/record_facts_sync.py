"""record_facts 동기화 — 백엔드 MySQL의 자물쇠 변경을 AI PostgreSQL에 반영한다 (백엔드 DB는 읽기만).

두 가지 방식을 조합해서 서로의 빈틈을 메운다.
- 최근 변경(sync_recent): 5분마다 최근 몇십 분 안에 저장된 자물쇠를 다시 읽어 넣고, 최근에
  삭제된 자물쇠를 지운다. 마지막으로 어디까지 읽었는지는 저장하지 않고 매번 겹쳐 읽어서
  놓치지 않는다. 같은 자물쇠를 두 번 넣어도 한 행만 유지되므로(멱등) 겹쳐 읽어도 안전하다.
- 전체 맞추기(reconcile): 하루 한 번 백엔드의 모든 자물쇠를 다시 넣고(수정도 반영), 백엔드에서
  사라졌는데 남아 있는 행을 지운다. 최근 변경이 놓친 것(오래된 자물쇠의 수정, 서버가 멈춰
  있던 동안의 변경, 완전 삭제)을 바로잡는다.

백엔드가 쓰는 시각(created_at, deleted_at)은 한국시간 값(시간대 없음)이라 "지금"도 한국시간으로 맞춰
비교한다. 읽기 전용 계정만 쓰고 백엔드 DB에는 어떤 것도 쓰지 않는다.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.repositories.record_fact_repository import RecordFactData, RecordFactRepository

logger = logging.getLogger("muro.record_facts_sync")

KST = timezone(timedelta(hours=9))  # 한국은 서머타임이 없어 고정 오프셋으로 충분하다

_COLUMNS = """
    r.record_id, r.user_id, r.created_at, r.weather_condition, r.temperature,
    r.mood_score, r.comment, p.legal_dong_name, p.legal_dong_code,
    p.latitude, p.longitude, m.title AS track_title, m.artist_name
"""
# 장소나 곡이 비어 있는 자물쇠도 빠뜨리지 않으려고 LEFT JOIN으로 읽는다
_FROM = """
    FROM records r
    LEFT JOIN places p ON r.place_id = p.place_id
    LEFT JOIN music_tracks m ON r.music_track_id = m.music_track_id
"""

# 삭제되지 않은 모든 자물쇠 (채우기·전체 맞추기)
SELECT_ACTIVE = text(
    f"""
    SELECT {_COLUMNS} {_FROM}
    WHERE r.deleted_at IS NULL AND r.record_id > :after_id
    ORDER BY r.record_id
    LIMIT :limit
    """
)
# 최근에 저장된 자물쇠 (최근 변경)
SELECT_RECENT = text(
    f"""
    SELECT {_COLUMNS} {_FROM}
    WHERE r.deleted_at IS NULL AND r.created_at >= :since AND r.record_id > :after_id
    ORDER BY r.record_id
    LIMIT :limit
    """
)
# 자물쇠 한 건 (저장 직후 바로 반영). 삭제된 자물쇠는 읽히지 않아서 삭제 뒤에 다시 살아나지 않는다
SELECT_ONE = text(
    f"""
    SELECT {_COLUMNS} {_FROM}
    WHERE r.deleted_at IS NULL AND r.record_id = :record_id
    """
)
# 최근에 삭제된 자물쇠의 ID
SELECT_DELETED_SINCE = text(
    """
    SELECT r.record_id
    FROM records r
    WHERE r.deleted_at IS NOT NULL AND r.deleted_at >= :since AND r.record_id > :after_id
    ORDER BY r.record_id
    LIMIT :limit
    """
)
# 주어진 ID 중 아직 삭제되지 않은 자물쇠의 ID
SELECT_STILL_ACTIVE = text(
    "SELECT record_id FROM records WHERE deleted_at IS NULL AND record_id IN :ids"
).bindparams(bindparam("ids", expanding=True))


@dataclass
class SyncCounts:
    upserted: int = 0
    deleted: int = 0
    with_comment: int = 0


def now_kst() -> datetime:
    """백엔드 created_at·deleted_at과 비교하는 지금 (한국시간, 시간대 없음)."""
    return datetime.now(KST).replace(tzinfo=None)


def fact_from_row(row, *, store_comment: bool) -> RecordFactData:
    """백엔드 MySQL의 한 행을 record_facts 한 행으로 바꾼다. 없는 값은 None으로 둔다."""
    comment = row.comment.strip() if row.comment and row.comment.strip() else None
    return RecordFactData(
        record_id=row.record_id,
        user_id=row.user_id,
        recorded_at=row.created_at,
        weather_condition=row.weather_condition,
        temperature=_to_float(row.temperature),
        mood_score=None if row.mood_score is None else int(row.mood_score),
        dong_name=row.legal_dong_name,
        dong_code=None if row.legal_dong_code is None else str(row.legal_dong_code),
        latitude=_to_float(row.latitude),
        longitude=_to_float(row.longitude),
        comment_text=comment if store_comment else None,
        track_title=row.track_title,
        artist_name=row.artist_name,
    )


def _to_float(value: object) -> float | None:
    # MySQL DECIMAL은 Decimal로 온다
    return None if value is None else float(value) if isinstance(value, Decimal) else value


async def copy_records(
    mysql: async_sessionmaker[AsyncSession],
    pg: async_sessionmaker[AsyncSession],
    *,
    statement,
    params: dict | None = None,
    batch_size: int,
    store_comment: bool,
    dry_run: bool = False,
    repository=RecordFactRepository,
) -> SyncCounts:
    """statement가 고른 자물쇠를 recordId 순서로 batch_size건씩 읽어 record_facts에 넣는다."""
    counts = SyncCounts()
    after_id = 0
    while True:
        async with mysql() as backend:
            rows = (
                await backend.execute(
                    statement, {**(params or {}), "after_id": after_id, "limit": batch_size}
                )
            ).all()
        if not rows:
            return counts
        after_id = rows[-1].record_id
        facts = [fact_from_row(row, store_comment=store_comment) for row in rows]
        counts.upserted += len(facts)
        counts.with_comment += sum(1 for f in facts if f.comment_text)
        if not dry_run:
            async with pg() as session:
                await repository(session).upsert_many(facts)


async def sync_one(
    mysql: async_sessionmaker[AsyncSession],
    pg: async_sessionmaker[AsyncSession],
    *,
    record_id: int,
    store_comment: bool,
    repository=RecordFactRepository,
) -> bool:
    """자물쇠 한 건을 백엔드에서 읽어 record_facts에 넣는다. 넣었으면 True.

    백엔드에 없거나 삭제된 자물쇠면 아무것도 하지 않는다 (False). 기본키로 한 행만 읽는다.
    """
    async with mysql() as backend:
        row = (await backend.execute(SELECT_ONE, {"record_id": record_id})).first()
    if row is None:
        return False
    async with pg() as session:
        await repository(session).upsert_many([fact_from_row(row, store_comment=store_comment)])
    return True


async def delete_deleted_since(
    mysql: async_sessionmaker[AsyncSession],
    pg: async_sessionmaker[AsyncSession],
    *,
    since: datetime,
    batch_size: int,
    repository=RecordFactRepository,
) -> int:
    """since 이후 백엔드에서 삭제된 자물쇠를 record_facts에서 지운다. 지운 후보 수를 돌려준다."""
    total = 0
    after_id = 0
    while True:
        async with mysql() as backend:
            rows = (
                await backend.execute(
                    SELECT_DELETED_SINCE,
                    {"since": since, "after_id": after_id, "limit": batch_size},
                )
            ).all()
        if not rows:
            return total
        after_id = rows[-1].record_id
        total += len(rows)
        async with pg() as session:
            await repository(session).delete_many([row.record_id for row in rows])


async def remove_orphans(
    mysql: async_sessionmaker[AsyncSession],
    pg: async_sessionmaker[AsyncSession],
    *,
    batch_size: int,
    repository=RecordFactRepository,
) -> int:
    """백엔드에는 없거나 삭제됐는데 record_facts에 남은 행을 지운다 (완전 삭제도 잡는다)."""
    removed = 0
    after_id = 0
    while True:
        async with pg() as session:
            ids = await repository(session).list_record_ids(after_id=after_id, limit=batch_size)
        if not ids:
            return removed
        after_id = ids[-1]
        async with mysql() as backend:
            alive = {
                r.record_id for r in (await backend.execute(SELECT_STILL_ACTIVE, {"ids": ids}))
            }
        orphans = [i for i in ids if i not in alive]
        if orphans:
            async with pg() as session:
                await repository(session).delete_many(orphans)
            removed += len(orphans)


async def sync_recent(
    mysql: async_sessionmaker[AsyncSession],
    pg: async_sessionmaker[AsyncSession],
    *,
    since: datetime,
    batch_size: int,
    store_comment: bool,
    repository=RecordFactRepository,
) -> SyncCounts:
    """최근 변경 반영: since 이후 저장된 자물쇠를 넣고, since 이후 삭제된 자물쇠를 지운다."""
    counts = await copy_records(
        mysql,
        pg,
        statement=SELECT_RECENT,
        params={"since": since},
        batch_size=batch_size,
        store_comment=store_comment,
        repository=repository,
    )
    counts.deleted = await delete_deleted_since(
        mysql, pg, since=since, batch_size=batch_size, repository=repository
    )
    return counts


async def reconcile(
    mysql: async_sessionmaker[AsyncSession],
    pg: async_sessionmaker[AsyncSession],
    *,
    batch_size: int,
    store_comment: bool,
    repository=RecordFactRepository,
) -> SyncCounts:
    """전체 맞추기: 백엔드의 모든 자물쇠를 다시 넣고(수정 반영), 사라진 자물쇠의 행을 지운다."""
    counts = await copy_records(
        mysql,
        pg,
        statement=SELECT_ACTIVE,
        batch_size=batch_size,
        store_comment=store_comment,
        repository=repository,
    )
    counts.deleted = await remove_orphans(mysql, pg, batch_size=batch_size, repository=repository)
    return counts
