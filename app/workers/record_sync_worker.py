"""기능4 자물쇠 동기화 — 5분마다 백엔드 DB에서 바뀐 자물쇠를 읽어 임베딩 큐에 넣는다 (pull).

- 새로 생기거나 바뀐 자물쇠 → 기존 임베딩 작업(embedding)으로 넣는다. 백엔드가 예전에
  POST /api/embeddings/generate로 보내던 것과 같은 내용이라 처리 코드는 그대로 쓴다.
- 삭제된 자물쇠 → 대기 중인 임베딩 작업을 취소하고 저장된 벡터를 지운다.
- 요청 본문에 없던 값(날씨·기온·기분·장소·곡명)도 같이 읽어 record_facts에 쓴다 — 기억 검색이
  백엔드 DB를 묻지 않고 AI DB에서 읽을 수 있게 새로 생기거나 바뀐 자물쇠마다 최신으로 맞춘다.
- 읽은 위치를 기억해서 다음엔 그 뒤부터만 읽는다. 처음엔 전체를 읽는다(백필).
  워커나 모델 서버가 한동안 멈춰도 다음에 밀린 만큼 따라잡는다.

읽기 자체는 가벼운 쿼리라 비용이 없고, 모델 호출은 바뀐 자물쇠 수만큼만 생긴다.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from app.clients.photo_signer import PhotoSigner, build_photo_signer
from app.config import get_settings
from app.db import mysql
from app.db.postgres import get_sessionmaker
from app.db.repositories.embedding_repository import EmbeddingRepository
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.record_fact_repository import RecordFactData, RecordFactRepository
from app.db.repositories.record_sync_repository import (
    BackendRecordReader,
    ChangedRecord,
    Cursor,
    SyncCursorStore,
)
from app.workers import embedding_worker

logger = logging.getLogger("muro.record_sync")

JOB_TYPE = "record_sync"


def slot_key(now: datetime) -> str:
    """5분 칸마다 작업 하나 — 같은 칸 안에서는 서버가 여러 대여도 한 번만 넣는다."""
    minutes = get_settings().record_sync_interval_minutes
    utc = now.astimezone(UTC)
    slot = utc.replace(minute=utc.minute - utc.minute % minutes, second=0, microsecond=0)
    return slot.strftime("%Y-%m-%dT%H:%M")


def job_fields(now: datetime) -> dict[str, Any]:
    return {
        "job_type": JOB_TYPE,
        "dedupe_key": slot_key(now),
        "payload": {},
        "max_attempts": 1,  # 실패하면 다음 5분 칸이 이어서 읽는다
    }


def embedding_payload(record: ChangedRecord, photo_url: str) -> dict[str, Any]:
    """백엔드가 예전에 보내 주던 POST /api/embeddings/generate 요청과 같은 모양."""
    return {
        "recordId": record.record_id,
        "userId": record.user_id,
        "photoUrl": photo_url,
        "track": {
            "title": record.title,
            "artistName": record.artist_name,
            "externalTrackId": record.external_track_id,
        },
        "comment": record.comment,
        "createdAt": record.created_at.isoformat(),
    }


async def handle(payload: dict[str, Any]) -> None:
    s = get_settings()
    if not s.backend_mysql_url:
        logger.warning("백엔드 DB 설정 없음 — 자물쇠 동기화 건너뜀")
        return
    signer = build_photo_signer(s.gcs_photo_bucket, s.photo_signed_url_minutes)
    async with mysql.get_sessionmaker()() as my, get_sessionmaker()() as pg:
        await sync(
            BackendRecordReader(my),
            SyncCursorStore(pg),
            JobRepository(pg),
            EmbeddingRepository(pg),
            signer,
            s.record_sync_batch_size,
            facts=RecordFactRepository(pg),
            store_comment=s.record_facts_store_comment,
        )


def fact_of(record: ChangedRecord, *, store_comment: bool) -> RecordFactData:
    """기억 검색용 record_facts 한 행. 코멘트 원문은 설정을 켰을 때만 담는다 (개인정보)."""
    comment = record.comment.strip() if record.comment and record.comment.strip() else None
    return RecordFactData(
        record_id=record.record_id,
        user_id=record.user_id,
        recorded_at=record.recorded_at_local or record.created_at.replace(tzinfo=None),
        weather_condition=record.weather_condition,
        temperature=record.temperature,
        mood_score=record.mood_score,
        dong_name=record.dong_name,
        dong_code=record.dong_code,
        latitude=record.latitude,
        longitude=record.longitude,
        comment_text=comment if store_comment else None,
        track_title=record.title or None,
        artist_name=record.artist_name or None,
        raw_record=_raw_without_comment(record, store_comment),
    )


def _raw_without_comment(record: ChangedRecord, store_comment: bool) -> dict[str, Any] | None:
    """원문 전체. 코멘트 저장을 껐다면 raw에서도 코멘트 칸을 뺀다 (설정이 서로 어긋나지 않게)."""
    if record.raw_record is None:
        return None
    if store_comment:
        return record.raw_record
    return {k: v for k, v in record.raw_record.items() if k != "comment"}


async def sync(
    reader,
    cursors,
    jobs,
    store,
    signer: PhotoSigner,
    batch_size: int,
    facts=None,
    store_comment: bool = False,
) -> int:
    """바뀐 자물쇠를 모두 처리하고 처리한 개수를 돌려준다. 한 건 처리할 때마다 위치를 저장해서
    중간에 실패해도 처리한 건 다시 하지 않는다."""
    cursor = await cursors.load()
    done = 0
    while True:
        records = await reader.changed_since(cursor, batch_size)
        for record in records:
            await _apply(record, jobs, store, signer, facts, store_comment)
            cursor = Cursor(record.updated_at, record.record_id)
            await cursors.save(cursor)
            done += 1
        if len(records) < batch_size:
            break
    if done:
        logger.info("자물쇠 동기화: %d건 처리 (마지막 %s)", done, cursor.updated_at)
    return done


async def _apply(
    record: ChangedRecord, jobs, store, signer: PhotoSigner, facts, store_comment: bool
) -> None:
    key = str(record.record_id)
    if record.deleted_at is not None:
        await jobs.cancel(job_type=embedding_worker.JOB_TYPE, dedupe_key=key)
        await store.delete(record.record_id)
        if facts is not None:
            await facts.delete_many([record.record_id])
        return
    if facts is not None:
        # 사진이 없어도 날씨·장소 같은 값은 기억 검색에 쓰이니 먼저 쓴다
        await facts.upsert_many([fact_of(record, store_comment=store_comment)])
    if not record.photo_key:
        # 자물쇠 사진은 필수라 정상이면 없을 수 없다 — 데이터 문제로 보고 건너뛴다
        logger.warning("사진 없는 자물쇠 %s — 임베딩 건너뜀", record.record_id)
        return
    try:
        photo_url = signer.sign(record.photo_key)
    except Exception:
        # 사진 주소를 못 만들어도(권한·버킷 설정·없는 파일) 다른 자물쇠의 동기화를 막지 않는다.
        # 이 자물쇠의 원문(record_facts)은 이미 저장했고, 임베딩은 자물쇠가 바뀔 때 다시 시도된다
        logger.warning(
            "자물쇠 %s 사진 주소 생성 실패 — 임베딩 건너뜀", record.record_id, exc_info=True
        )
        return
    await jobs.enqueue(
        job_type=embedding_worker.JOB_TYPE,
        dedupe_key=key,
        payload=embedding_payload(record, photo_url),
        max_attempts=get_settings().job_max_attempts,
    )


async def schedule(now: datetime) -> None:
    async with get_sessionmaker()() as session:
        await JobRepository(session).enqueue_if_absent(**job_fields(now))
