"""가수 이름 별칭(영어↔한글) 만들기 — 주기마다 아직 변환하지 않은 가수만 LLM으로 변환해 저장한다.

- record_facts에 저장된 가수 이름("A, B"는 한 명씩)에서 artist_aliases에 없는 가수만 고른다.
  새 가수가 없으면 쿼리 두 번으로 끝나고 LLM은 부르지 않는다. 처음 켜면 있는 가수를 모두 채운다.
- LLM이 모른다고 답한 가수는 빈 목록으로 저장해서 다시 묻지 않는다. LLM 호출이 실패한 가수는
  저장하지 않아서 다음 주기에 다시 시도한다.
- 한 번에 최대 artist_alias_max_per_run명까지만 변환한다 (비용 상한). 나머지는 다음 주기에.
- 사람이 서버에서 실행할 필요가 없다: 워커 루프가 주기마다 작업을 하나씩 큐에 넣는다.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from app.components.artist_alias_tagger import ArtistAliasTagger, needs_alias
from app.components.track_lexicon import artist_names, compact
from app.config import get_settings
from app.db.postgres import get_sessionmaker
from app.db.repositories.artist_alias_repository import ArtistAliasRepository
from app.db.repositories.job_repository import JobRepository
from app.dependencies import get_clients

logger = logging.getLogger("muro.artist_alias")

JOB_TYPE = "artist_alias"

SOURCE_LIMIT = 20000  # record_facts에서 읽는 가수 칸 값의 최대 개수
SAVE_CHUNK = 20  # 이만큼 변환할 때마다 저장한다 (워커가 중간에 멈춰도 한 덩이만 잃는다)
MAX_KEY_LENGTH = 200  # artist_aliases.name_key 길이


def slot_key(now: datetime) -> str:
    """주기 칸마다 작업 하나 — 같은 칸 안에서는 서버가 여러 대여도 한 번만 넣는다."""
    minutes = max(1, get_settings().artist_alias_interval_minutes)
    return str(int(now.timestamp() // 60 // minutes))


def job_fields(now: datetime) -> dict[str, Any]:
    return {
        "job_type": JOB_TYPE,
        "dedupe_key": slot_key(now),
        "payload": {},
        "max_attempts": 1,  # 실패하면 다음 주기가 이어서 한다
    }


async def schedule(now: datetime) -> None:
    async with get_sessionmaker()() as session:
        await JobRepository(session).enqueue_if_absent(**job_fields(now))


@dataclass(frozen=True)
class RefreshResult:
    converted: int  # 이번에 저장한 가수 수 (모른다고 답한 것 포함)
    failed: int  # LLM 호출이 실패해 저장하지 않은 가수 수
    remaining: int  # 상한 때문에 다음 주기로 넘긴 가수 수


async def handle(payload: dict[str, Any]) -> None:
    s = get_settings()
    if not s.artist_alias_enabled or s.ai_client_mode == "fake":
        return
    tagger = ArtistAliasTagger(get_clients().llm_knowledge)
    async with get_sessionmaker()() as session:
        await refresh(
            ArtistAliasRepository(session),
            tagger,
            concurrency=s.artist_alias_concurrency,
            max_per_run=s.artist_alias_max_per_run,
        )


async def refresh(
    repository, tagger, *, concurrency: int, max_per_run: int, chunk_size: int = SAVE_CHUNK
) -> RefreshResult:
    """변환하지 않은 가수를 최대 max_per_run명 변환해 저장한다."""
    wanted: dict[str, str] = {}
    for column in await repository.source_artist_names(SOURCE_LIMIT):
        for name in artist_names(column):
            key = compact(name)
            if key and len(key) <= MAX_KEY_LENGTH and needs_alias(name):
                wanted.setdefault(key, name)

    known = await repository.known_keys(tagger.version)
    todo = [(key, name) for key, name in wanted.items() if key not in known]
    remaining = max(0, len(todo) - max_per_run)
    todo = todo[:max_per_run]
    if not todo:
        return RefreshResult(converted=0, failed=0, remaining=0)

    gate = asyncio.Semaphore(max(1, concurrency))

    async def convert(key: str, name: str) -> tuple[str, str, tuple[str, ...] | None]:
        async with gate:
            try:
                return key, name, await tagger.aliases(name)
            except Exception:
                logger.warning(
                    "가수 별칭 변환 실패 (%s) — 다음 주기에 다시 시도", name, exc_info=True
                )
                return key, name, None

    converted = failed = 0
    for start in range(0, len(todo), chunk_size):
        results = await asyncio.gather(
            *(convert(k, n) for k, n in todo[start : start + chunk_size])
        )
        for key, name, aliases in results:
            if aliases is None:
                failed += 1
                continue
            await repository.save(
                name_key=key, artist_name=name, aliases=aliases, version=tagger.version
            )
            converted += 1
    logger.info("가수 별칭: %d명 저장, %d명 실패, %d명은 다음 주기로", converted, failed, remaining)
    return RefreshResult(converted=converted, failed=failed, remaining=remaining)
