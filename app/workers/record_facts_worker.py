"""record_facts 동기화 작업 — 백엔드 MySQL의 자물쇠 변경을 AI PostgreSQL에 반영한다.

두 종류의 작업을 큐에 넣는다 (동기화 방식은 app/components/record_facts_sync.py).
- recent: 5분마다. 최근 변경(저장·삭제)만 반영한다.
- full: 하루 한 번(설정한 시각 이후 첫 루프). 전체를 맞추고 사라진 자물쇠의 행을 지운다.

작업 이름(dedupe_key)을 시간 구간으로 만들어서 서버가 여러 대여도 구간마다 작업이 하나만
큐에 들어간다 (기능6 매일 배치와 같은 방식). 끝난 작업은 이틀 뒤에 지워서 표가 커지지 않게 한다.
설정(RECORD_FACTS_SYNC_ENABLED)을 켜고 BACKEND_MYSQL_URL이 있을 때만 동작한다.
"""

import logging
from datetime import datetime, timedelta
from typing import Any

from app.components.record_facts_sync import KST, now_kst, reconcile, sync_one, sync_recent
from app.config import get_settings
from app.db import mysql, postgres
from app.db.repositories.job_repository import JobRepository

logger = logging.getLogger("muro.record_facts_worker")

JOB_TYPE = "record_facts_sync"
MAX_ATTEMPTS = 3
BATCH_SIZE = 500
KEEP_FINISHED_HOURS = 48


def enabled() -> bool:
    s = get_settings()
    return bool(s.record_facts_sync_enabled and s.backend_mysql_url)


def recent_key(now: datetime) -> str:
    """5분(설정값) 구간 이름. 같은 구간이면 같은 이름이라 한 번만 큐에 들어간다."""
    interval = max(get_settings().record_facts_sync_interval_seconds, 1)
    bucket = int(now.timestamp()) // interval * interval
    return "recent:" + datetime.fromtimestamp(bucket, KST).strftime("%Y-%m-%dT%H:%M")


def full_key(now: datetime) -> str | None:
    """전체 맞추기를 할 시각(한국시간)이 지났으면 오늘 날짜 이름, 아니면 None."""
    local = now.astimezone(KST)
    if local.hour < get_settings().record_facts_full_sync_hour:
        return None
    return f"full:{local.date().isoformat()}"


def job_fields(mode: str, key: str) -> dict[str, Any]:
    return {
        "job_type": JOB_TYPE,
        "dedupe_key": key,
        "payload": {"mode": mode},
        "max_attempts": MAX_ATTEMPTS,
    }


async def enqueue_due(now: datetime, last: dict[str, str]) -> None:
    """지금 구간의 작업을 큐에 넣는다. last에 넣은 이름을 적어 두어 루프마다 DB를 치지 않는다."""
    if not enabled():
        return
    wanted = {"recent": recent_key(now), "full": full_key(now)}
    for mode, key in wanted.items():
        if key is None or last.get(mode) == key:
            continue
        async with postgres.get_sessionmaker()() as session:
            if await JobRepository(session).enqueue_if_absent(**job_fields(mode, key)):
                logger.info("record_facts 동기화 예약: %s", key)
        last[mode] = key


async def sync_record(record_id: int) -> None:
    """자물쇠가 저장되어 기능4 작업이 시작될 때 그 한 건의 조건 값을 바로 반영한다.

    5분 동기화를 기다리지 않아도 새 자물쇠가 조건 검색에 나온다. 부가 작업이라서 꺼져 있으면
    아무것도 안 하고, 실패해도 예외를 밖으로 내보내지 않는다 (임베딩 작업이 실패하지 않게).
    못 읽은 자물쇠는 5분 동기화가 다시 잡는다.
    """
    if not enabled():
        return
    try:
        await sync_one(
            mysql.get_sessionmaker(),
            postgres.get_sessionmaker(),
            record_id=record_id,
            store_comment=get_settings().record_facts_store_comment,
        )
    except Exception:
        logger.warning(
            "record_facts 한 건 반영 실패 (record %s) — 동기화가 다시 잡는다",
            record_id,
            exc_info=True,
        )


async def handle(payload: dict[str, Any]) -> None:
    if not enabled():
        logger.info("record_facts 동기화가 꺼져 있어 건너뜀")
        return
    s = get_settings()
    mysql_maker, pg_maker = mysql.get_sessionmaker(), postgres.get_sessionmaker()
    mode = payload.get("mode")
    if mode == "full":
        counts = await reconcile(
            mysql_maker,
            pg_maker,
            batch_size=BATCH_SIZE,
            store_comment=s.record_facts_store_comment,
        )
    else:
        since = now_kst() - timedelta(minutes=s.record_facts_sync_lookback_minutes)
        counts = await sync_recent(
            mysql_maker,
            pg_maker,
            since=since,
            batch_size=BATCH_SIZE,
            store_comment=s.record_facts_store_comment,
        )
    logger.info(
        "record_facts 동기화(%s): 반영 %d건, 삭제 %d건", mode, counts.upserted, counts.deleted
    )
    async with pg_maker() as session:
        await JobRepository(session).delete_finished(
            job_type=JOB_TYPE, older_than_hours=KEEP_FINISHED_HOURS
        )
