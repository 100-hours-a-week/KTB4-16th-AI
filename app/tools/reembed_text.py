"""텍스트 벡터 재임베딩 도구 — 마이그레이션 0010(OpenAI → KURE) 뒤에 비어 있는 벡터를 다시 채운다.

실행 (배포한 컨테이너 안에서, 임베딩 서비스가 떠 있어야 한다):
    python -m app.tools.reembed_text --dry-run   # 몇 건이 비어 있는지만 센다
    python -m app.tools.reembed_text             # 실제로 채운다

무엇을 채우나
- track_moods.mood_embedding: 저장된 무드 글(mood_text)을 다시 임베딩. LLM은 부르지 않는다.
  mood_version의 임베딩 모델 자리도 새 모델로 바꿔서 캐시로 다시 쓰이게 한다
  (LLM 모델·프롬프트 버전이 지금과 다른 행은 어차피 새로 만들어져서 값만 채우고 버전은 둔다).
- record_embeddings.music_embedding / comment_embedding: 곡 무드 글은 AI DB에 있고, 코멘트 원문은
  AI DB에 없어서 백엔드 MySQL(읽기 전용)에서 읽는다. model_versions의 text·moodVersion도 갱신한다.

안전장치
- 비어 있는 행(music_embedding / mood_embedding이 NULL)만 고른다. 여러 번 돌려도 같은 결과이고,
  중간에 멈춰도 다시 돌리면 이어서 한다.
- 갱신할 때 "아직 비어 있을 때만" 바꾼다. 도구가 도는 동안 워커가 새로 저장한 행을 덮어쓰지 않는다.
- 사진 벡터(CLIP)와 외부 LLM·Replicate는 쓰지 않는다. 임베딩은 같은 컨테이너의 KURE 서비스만 부른다.
"""

import argparse
import asyncio
import logging
from dataclasses import dataclass

from sqlalchemy import bindparam, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.components.embedder import Embedder
from app.components.music_mood_tagger import PROMPT_VERSION
from app.components.track_mood_resolver import build_mood_version
from app.config import get_settings
from app.db.models import RecordEmbedding, TrackMood

logger = logging.getLogger("muro.reembed")

DEFAULT_BATCH_SIZE = 64


@dataclass
class Counts:
    track_moods: int = 0
    records: int = 0
    comments: int = 0
    skipped_blank: int = 0


def next_mood_version(old_version: str, llm_model: str, embedder_model: str) -> str:
    """무드 캐시 버전의 임베딩 모델 자리를 새 모델로 바꾼다.

    LLM 모델·프롬프트 버전이 지금과 같은 행만 새 버전으로 올린다. 다르면 지금 설정으로는 어차피
    캐시 미스라서 버전을 그대로 둔다 (잘못 맞는 캐시가 되지 않게).
    """
    head, _, _ = old_version.rpartition("|")
    if head != f"{llm_model}|{PROMPT_VERSION}":
        return old_version
    return build_mood_version(llm_model, embedder_model)


def next_record_versions(
    old: dict[str, object] | None, embedder_model: str, mood_version: str
) -> dict[str, object]:
    versions = dict(old or {})
    versions["text"] = embedder_model
    versions["moodVersion"] = mood_version
    return versions


async def reembed_track_moods(
    pg: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    *,
    llm_model: str,
    batch_size: int,
    dry_run: bool,
    counts: Counts,
) -> None:
    last_id = ""
    while True:
        async with pg() as session:
            rows = (
                await session.execute(
                    select(TrackMood.external_track_id, TrackMood.mood_text, TrackMood.mood_version)
                    .where(
                        TrackMood.mood_embedding.is_(None), TrackMood.external_track_id > last_id
                    )
                    .order_by(TrackMood.external_track_id)
                    .limit(batch_size)
                )
            ).all()
            if not rows:
                return
            last_id = rows[-1].external_track_id
            counts.track_moods += len(rows)
            if dry_run:
                continue

            usable = [r for r in rows if r.mood_text and r.mood_text.strip()]
            counts.skipped_blank += len(rows) - len(usable)
            vectors = await embedder.embed_batch([r.mood_text for r in usable]) if usable else []
            for row, vector in zip(usable, vectors, strict=True):
                await session.execute(
                    update(TrackMood)
                    .where(
                        TrackMood.external_track_id == row.external_track_id,
                        TrackMood.mood_embedding.is_(None),
                    )
                    .values(
                        mood_embedding=vector,
                        mood_version=next_mood_version(row.mood_version, llm_model, embedder.model),
                    )
                )
            await session.commit()
        logger.info("track_moods %d건 처리", counts.track_moods)


async def _read_comments(mysql: AsyncSession, record_ids: list[int]) -> dict[int, str]:
    rows = await mysql.execute(
        text(
            "SELECT record_id, comment FROM records "
            "WHERE record_id IN :record_ids AND deleted_at IS NULL"
        ).bindparams(bindparam("record_ids", expanding=True)),
        {"record_ids": record_ids},
    )
    return {r.record_id: r.comment for r in rows if r.comment and r.comment.strip()}


async def reembed_records(
    pg: async_sessionmaker[AsyncSession],
    mysql: async_sessionmaker[AsyncSession],
    embedder: Embedder,
    *,
    llm_model: str,
    batch_size: int,
    dry_run: bool,
    counts: Counts,
) -> None:
    last_id = 0
    while True:
        async with pg() as session:
            rows = (
                await session.execute(
                    select(
                        RecordEmbedding.record_id,
                        RecordEmbedding.music_mood_text,
                        RecordEmbedding.model_versions,
                    )
                    .where(
                        RecordEmbedding.music_embedding.is_(None),
                        RecordEmbedding.record_id > last_id,
                    )
                    .order_by(RecordEmbedding.record_id)
                    .limit(batch_size)
                )
            ).all()
            if not rows:
                return
            last_id = rows[-1].record_id
            counts.records += len(rows)
            if dry_run:
                continue

            usable = [r for r in rows if r.music_mood_text and r.music_mood_text.strip()]
            counts.skipped_blank += len(rows) - len(usable)
            if not usable:
                continue
            async with mysql() as backend:
                comments = await _read_comments(backend, [r.record_id for r in usable])
            music_vectors = await embedder.embed_batch([r.music_mood_text for r in usable])
            comment_ids = [r.record_id for r in usable if r.record_id in comments]
            comment_vectors = dict(
                zip(
                    comment_ids,
                    await embedder.embed_batch([comments[i] for i in comment_ids])
                    if comment_ids
                    else [],
                    strict=True,
                )
            )
            counts.comments += len(comment_vectors)
            for row, music_vector in zip(usable, music_vectors, strict=True):
                old = row.model_versions or {}
                mood_version = next_mood_version(
                    str(old.get("moodVersion", "")), llm_model, embedder.model
                )
                await session.execute(
                    update(RecordEmbedding)
                    .where(
                        RecordEmbedding.record_id == row.record_id,
                        RecordEmbedding.music_embedding.is_(None),
                    )
                    .values(
                        music_embedding=music_vector,
                        comment_embedding=comment_vectors.get(row.record_id),
                        model_versions=next_record_versions(old, embedder.model, mood_version),
                    )
                )
            await session.commit()
        logger.info("record_embeddings %d건 처리", counts.records)


async def run(*, batch_size: int, dry_run: bool) -> Counts:
    # 무겁거나 키가 필요한 의존성은 실행할 때만 불러온다
    from app.db import mysql as mysql_db
    from app.db import postgres as pg_db
    from app.dependencies import get_clients

    settings = get_settings()
    if settings.text_embedding_provider != "kure":
        raise SystemExit("TEXT_EMBEDDING_PROVIDER가 kure일 때만 쓸 수 있어요.")
    if not settings.backend_mysql_url:
        raise SystemExit("BACKEND_MYSQL_URL이 없어서 코멘트 원문을 읽을 수 없어요.")

    clients = get_clients()
    embedder = Embedder(clients.embedding)
    await embedder.embed("연결 확인")  # 임베딩 서비스가 안 떠 있으면 여기서 바로 멈춘다

    counts = Counts()
    kwargs = {
        "embedder": embedder,
        "llm_model": clients.llm_knowledge.model,
        "batch_size": batch_size,
        "dry_run": dry_run,
        "counts": counts,
    }
    await reembed_track_moods(pg_db.get_sessionmaker(), **kwargs)
    await reembed_records(pg_db.get_sessionmaker(), mysql_db.get_sessionmaker(), **kwargs)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="비어 있는 텍스트 벡터를 KURE로 다시 채운다")
    parser.add_argument("--dry-run", action="store_true", help="채우지 않고 대상 건수만 센다")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    counts = asyncio.run(run(batch_size=args.batch_size, dry_run=args.dry_run))
    label = "대상" if args.dry_run else "처리"
    print(
        f"{label}: track_moods {counts.track_moods}건, record_embeddings {counts.records}건"
        + (
            ""
            if args.dry_run
            else f" (코멘트 {counts.comments}건, 글이 비어 건너뜀 {counts.skipped_blank}건)"
        )
    )


if __name__ == "__main__":
    main()
