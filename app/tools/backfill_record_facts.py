"""자물쇠 조건 값 채우기 도구 — 백엔드 MySQL의 자물쇠 정보를 AI PostgreSQL(record_facts)로 옮긴다.

마이그레이션 0011이 표만 만들기 때문에, 이미 저장된 자물쇠의 값은 이 도구가 한 번 채운다.
기억 검색이 읽는 곳을 postgres로 바꾸기(MEMORY_RECORD_SOURCE) 전에 실행해야 한다.

실행 (배포한 컨테이너 안에서, BACKEND_MYSQL_URL이 있어야 한다):
    python -m app.tools.backfill_record_facts --dry-run   # 몇 건을 옮길지만 센다
    python -m app.tools.backfill_record_facts             # 실제로 옮긴다
    python -m app.tools.backfill_record_facts --with-comments   # 코멘트 원문도 저장 (개인정보)

무엇을 옮기나
- records(날짜·날씨·기온·기분), places(법정동 이름·코드·좌표), music_tracks(곡 제목·가수).
  앨범 이미지·링크 같은 카드 정보는 읽지 않는다.
- 삭제된 자물쇠(deleted_at)는 옮기지 않는다.
- 코멘트 원문은 기본으로 저장하지 않는다 (설정 RECORD_FACTS_STORE_COMMENT 또는 --with-comments).

안전장치
- 백엔드 DB는 읽기만 한다. AI DB에는 같은 recordId가 이미 있으면 덮어써서(멱등) 여러 번 돌려도
  같은 결과이고, 중간에 멈춰도 다시 돌리면 된다 (처음부터 다시 훑지만 결과는 같다).
- recordId 순서로 한 번에 batch-size건씩 읽어서 메모리를 적게 쓴다.
- 장소나 곡이 비어 있는 자물쇠도 빠뜨리지 않고 옮긴다 (그 칸만 비어 있다).
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import get_settings
from app.db.repositories.record_fact_repository import RecordFactData, RecordFactRepository

logger = logging.getLogger("muro.backfill_facts")

DEFAULT_BATCH_SIZE = 500

_SELECT = text(
    """
    SELECT r.record_id, r.user_id, r.created_at, r.weather_condition, r.temperature,
           r.mood_score, r.comment, p.legal_dong_name, p.legal_dong_code,
           p.latitude, p.longitude, m.title AS track_title, m.artist_name
    FROM records r
    LEFT JOIN places p ON r.place_id = p.place_id
    LEFT JOIN music_tracks m ON r.music_track_id = m.music_track_id
    WHERE r.deleted_at IS NULL AND r.record_id > :after_id
    ORDER BY r.record_id
    LIMIT :limit
    """
)


@dataclass
class Counts:
    records: int = 0
    with_comment: int = 0


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


async def backfill(
    mysql: async_sessionmaker[AsyncSession],
    pg: async_sessionmaker[AsyncSession],
    *,
    batch_size: int,
    dry_run: bool,
    store_comment: bool,
    repository=RecordFactRepository,
) -> Counts:
    counts = Counts()
    after_id = 0
    while True:
        async with mysql() as backend:
            rows = (
                await backend.execute(_SELECT, {"after_id": after_id, "limit": batch_size})
            ).all()
        if not rows:
            return counts
        after_id = rows[-1].record_id
        facts = [fact_from_row(row, store_comment=store_comment) for row in rows]
        counts.records += len(facts)
        counts.with_comment += sum(1 for f in facts if f.comment_text)
        if not dry_run:
            async with pg() as session:
                await repository(session).upsert_many(facts)
        logger.info("record_facts %d건 처리", counts.records)


async def run(*, batch_size: int, dry_run: bool, store_comment: bool) -> Counts:
    # 접속 설정이 필요한 의존성은 실행할 때만 불러온다
    from app.db import mysql as mysql_db
    from app.db import postgres as pg_db

    if not get_settings().backend_mysql_url:
        raise SystemExit("BACKEND_MYSQL_URL이 없어서 백엔드 자물쇠를 읽을 수 없어요.")
    return await backfill(
        mysql_db.get_sessionmaker(),
        pg_db.get_sessionmaker(),
        batch_size=batch_size,
        dry_run=dry_run,
        store_comment=store_comment,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="백엔드 자물쇠 조건 값을 record_facts로 옮긴다")
    parser.add_argument("--dry-run", action="store_true", help="옮기지 않고 대상 건수만 센다")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--with-comments", action="store_true", help="코멘트 원문도 저장한다 (개인정보)"
    )
    args = parser.parse_args()

    store_comment = args.with_comments or get_settings().record_facts_store_comment
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    counts = asyncio.run(
        run(batch_size=args.batch_size, dry_run=args.dry_run, store_comment=store_comment)
    )
    label = "대상" if args.dry_run else "처리"
    saved = "함" if store_comment else "안 함"
    print(
        f"{label}: record_facts {counts.records}건"
        f" (코멘트 원문 저장 {saved}, 코멘트가 있는 자물쇠 {counts.with_comment}건)"
    )


if __name__ == "__main__":
    main()
