"""기능6 매일 배치 — 다음 날 쓸 사용자 그룹(music_genre)과 그룹별 오늘의 질문을 AI DB에 저장.

워커가 매일 시작 시각(기본 23:00 KST)에 작업을 한 번 넣고, 실패하면 큐의 지수 백오프
(30초, 1분, 2분, 4분, 8분)로 마감(23:40) 전에 다시 돌린다. 백엔드는 마감 이후에 뷰를 읽는다.
다시 돌려도 안전하다 — 그룹은 덮어쓰고, 이미 질문이 있는 날짜·그룹은 건너뛴다.
"""

import logging
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from app.components.music_genre_assigner import assign_genres
from app.components.music_tags import COMMON_GROUP
from app.components.question_generator import PROMPT_VERSION, QuestionGenerator
from app.config import get_settings
from app.db import mysql
from app.db.postgres import get_sessionmaker
from app.db.repositories.balance_daily_repository import BalanceDailyRepository
from app.db.repositories.job_repository import JobRepository

logger = logging.getLogger("muro.balance_daily")

JOB_TYPE = "balance_daily"
MAX_ATTEMPTS = 6  # 백오프 합계 약 15분 — 23:00에 시작하면 마감 안에 끝난다
CANDIDATES_PER_GROUP = 3  # 하나를 고르되, 최근에 낸 질문과 겹치면 다음 후보로
KST = timezone(timedelta(hours=9))  # 한국은 서머타임이 없어 고정 오프셋으로 충분


class GroupFailedError(RuntimeError):
    """일부 그룹의 질문을 못 만듦 — 재시도하면 빈 그룹만 다시 채운다."""


def _clock(value: str) -> time:
    hour, minute = value.split(":")
    return time(int(hour), int(minute))


def due_date(now: datetime) -> date | None:
    """지금이 배치 시간(시작~마감)이면 계산할 날짜(내일), 아니면 None."""
    s = get_settings()
    local = now.astimezone(KST)
    if _clock(s.balance_daily_start) <= local.time() < _clock(s.balance_daily_deadline):
        return local.date() + timedelta(days=1)
    return None


def job_fields(target: date) -> dict[str, Any]:
    return {
        "job_type": JOB_TYPE,
        "dedupe_key": target.isoformat(),
        "payload": {"date": target.isoformat()},
        "max_attempts": MAX_ATTEMPTS,
    }


def tomorrow(now: datetime) -> date:
    return now.astimezone(KST).date() + timedelta(days=1)


async def enqueue_daily(target: date) -> None:
    """하루 한 번만 넣는다 (서버가 2대여도 큐의 유니크 키로 한 번)."""
    async with get_sessionmaker()() as session:
        if await JobRepository(session).enqueue_if_absent(**job_fields(target)):
            logger.info("기능6 매일 배치 예약: %s", target)


async def handle(payload: dict[str, Any]) -> None:
    # 늦게 import — dependencies가 서비스·클라이언트를 모두 불러와서 워커 시작이 무거워지지 않게
    from app.dependencies import get_clients, get_genre_stats_source

    s = get_settings()
    target = date.fromisoformat(payload["date"])
    deadline = datetime.combine(target - timedelta(days=1), _clock(s.balance_daily_deadline), KST)
    if datetime.now(KST) > deadline:
        logger.warning("기능6 배치가 마감(%s)을 넘겨 실행됨 — 계속 진행", deadline)

    async with get_sessionmaker()() as pg:
        if s.backend_mysql_url:
            async with mysql.get_sessionmaker()() as my:
                groups = await _assign(BalanceDailyRepository(my, pg), s.balance_min_group_size)
        else:
            logger.warning("백엔드 DB 설정 없음 — 그룹 계산 없이 공통 질문만 만든다")
            groups = {COMMON_GROUP}

        repo = BalanceDailyRepository(None, pg)
        generator = QuestionGenerator(get_clients().llm_knowledge)
        stats_source = get_genre_stats_source()
        used = await repo.recent_questions(target)
        failed: list[str] = []
        for offset in range(s.balance_question_days_ahead):
            day = target + timedelta(days=offset)
            for genre in sorted(groups - await repo.filled_groups(day)):
                try:
                    await _fill(repo, generator, stats_source, day, genre, used)
                except Exception:
                    logger.exception("질문 생성 실패: %s %s", day, genre)
                    failed.append(f"{day}/{genre}")
    if failed:
        raise GroupFailedError(f"질문을 못 만든 그룹: {', '.join(failed)}")
    logger.info("기능6 배치 완료: %s, 그룹 %d개", target, len(groups))


async def _assign(repo: BalanceDailyRepository, min_group_size: int) -> set[str]:
    saved = await repo.saved_tracks()
    genre_of = await repo.genre_of({t.external_track_id for t in saved})
    assignments = assign_genres(saved, genre_of, min_group_size)
    await repo.save_assignments(assignments)
    logger.info("사용자 그룹 %d명 저장", len(assignments))
    return {a.music_genre for a in assignments.values()} | {COMMON_GROUP}


async def _fill(repo, generator, stats_source, day: date, genre: str, used: set[str]) -> None:
    stats = await stats_source.get(genre)
    candidates = await generator.generate(genre, stats, CANDIDATES_PER_GROUP)
    fresh = [c for c in candidates if c[0] not in used]
    if not fresh:
        raise GroupFailedError(
            f"{genre}: 새 질문 없음 (후보 {len(candidates)}개 모두 최근 질문과 겹침)"
        )
    question, option_a, option_b = fresh[0]
    model = f"{generator.model}|{PROMPT_VERSION}"
    if await repo.add_question(
        question_date=day,
        music_genre=genre,
        question=question,
        option_a=option_a,
        option_b=option_b,
        model_version=model,
    ):
        used.add(question)
