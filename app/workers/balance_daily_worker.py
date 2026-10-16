"""기능6 매일 배치 — 다음 날 쓸 사용자 그룹(music_genre)과 그룹별 오늘의 질문을 AI DB에 저장.

워커가 매일 시작 시각(기본 23:00 KST)에 작업을 한 번 넣고, 실패하면 큐의 지수 백오프
(30초, 1분, 2분, 4분, 8분)로 마감(23:40) 전에 다시 돌린다. 백엔드는 마감 이후에 뷰를 읽는다.
다시 돌려도 안전하다 — 그룹은 덮어쓰고, 이미 질문이 있는 날짜·그룹은 건너뛴다.

2026-10-16: 그날 사용자가 있는 그룹에만 질문을 만들면, 사람이 없는 장르는 질문이 아예
없어서 preferred_genres로 그 장르에 들어간 사용자가 투표를 못 한다. 그래서 사용자 수와
상관없이 13개 장르 + 공통, MUSIC_GENRE_GROUPS 전부에 매일 질문을 만든다(사용자가 없는
장르는 question_generator가 가수 이름 없이 일반적인 질문을 만든다).
LLM이 CANDIDATES_PER_GROUP개를 내면 1개만 승인 질문으로 쓰고 나머지는 '예비 질문'으로
쌓아 둔다(status='spare'). 그날 새 질문을 하나도 못 만들면(형식 깨짐·최근 질문과 전부
겹침) 쌓아 둔 예비 질문을 꺼내 쓴다 — LLM을 다시 부르지 않고도 그룹마다 질문이 빈 날이
없게 한다. 예비도 없으면 그 그룹만 실패로 보고 재시도(백오프)에 맡긴다.

2026-10-16: 그룹마다 먼저 곡 대결(song_battle_generator)을 시도한다. 곡 풀에서 체급
(popularity)이 비슷한 두 곡을 직접 뽑아 보여주는 유형이라 LLM 비용이 0이고 지어낼
염려가 없다. 그 장르 곡 풀이 얇아서 쓸 짝이 없으면(None) 기존 LLM 취향 질문으로
넘어간다. "체급"은 지금은 TJ 차트 개월 수(popularity)이고, 나중에 유튜브 조회수로
바꾸려면 곡 풀의 popularity 값만 바꾸면 되고 이 워커는 그대로 쓴다.
"""

import logging
import random
from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from app.components.music_genre_assigner import assign_genres
from app.components.music_tags import MUSIC_GENRE_GROUPS
from app.components.question_generator import PROMPT_VERSION, QuestionGenerator
from app.components.song_battle_generator import PROMPT_VERSION as BATTLE_PROMPT_VERSION
from app.components.song_battle_generator import generate_battle
from app.components.song_pool import SongPool
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
    from app.dependencies import get_clients, get_genre_stats_source, get_song_pool

    s = get_settings()
    target = date.fromisoformat(payload["date"])
    deadline = datetime.combine(target - timedelta(days=1), _clock(s.balance_daily_deadline), KST)
    if datetime.now(KST) > deadline:
        logger.warning("기능6 배치가 마감(%s)을 넘겨 실행됨 — 계속 진행", deadline)

    async with get_sessionmaker()() as pg:
        if s.backend_mysql_url:
            async with mysql.get_sessionmaker()() as my:
                await _assign(BalanceDailyRepository(my, pg), s.balance_min_group_size)
        else:
            logger.warning("백엔드 DB 설정 없음 — 그룹 계산 없이 질문만 만든다")

        repo = BalanceDailyRepository(None, pg)
        generator = QuestionGenerator(get_clients().llm_knowledge)
        stats_source = get_genre_stats_source()
        pool = get_song_pool()
        used = await repo.recent_questions(target)
        battle_used_pairs = await repo.recent_battle_pairs(target)
        battle_rng = random.Random()
        failed: list[str] = []
        for offset in range(s.balance_question_days_ahead):
            day = target + timedelta(days=offset)
            for genre in sorted(MUSIC_GENRE_GROUPS - await repo.filled_groups(day)):
                try:
                    await _fill(
                        repo,
                        generator,
                        stats_source,
                        day,
                        genre,
                        used,
                        pool,
                        battle_used_pairs,
                        battle_rng,
                    )
                except Exception:
                    logger.exception("질문 생성 실패: %s %s", day, genre)
                    failed.append(f"{day}/{genre}")
    if failed:
        raise GroupFailedError(f"질문을 못 만든 그룹: {', '.join(failed)}")
    logger.info("기능6 배치 완료: %s, 그룹 %d개", target, len(MUSIC_GENRE_GROUPS))


async def _assign(repo: BalanceDailyRepository, min_group_size: int) -> None:
    saved = await repo.saved_tracks()
    genre_of = await repo.genre_of({t.external_track_id for t in saved})
    assignments = assign_genres(saved, genre_of, min_group_size)
    await repo.save_assignments(assignments)
    logger.info("사용자 그룹 %d명 저장", len(assignments))


async def _fill(
    repo,
    generator,
    stats_source,
    day: date,
    genre: str,
    used: set[str],
    pool: SongPool | None,
    battle_used_pairs: set[frozenset[str]],
    battle_rng: random.Random,
) -> None:
    if pool is not None:
        battle = generate_battle(genre, pool, battle_used_pairs, battle_rng)
        if battle is not None:
            if await repo.add_question(
                question_date=day,
                music_genre=genre,
                question=battle.question,
                option_a=battle.option_a,
                option_b=battle.option_b,
                model_version=f"song-battle|{BATTLE_PROMPT_VERSION}",
            ):
                battle_used_pairs.add(battle.pair_key)
            return

    stats = await stats_source.get(genre)
    candidates = await generator.generate(genre, stats, CANDIDATES_PER_GROUP)
    fresh = [c for c in candidates if c[0] not in used]
    model = f"{generator.model}|{PROMPT_VERSION}"

    if not fresh:
        # 새 질문을 하나도 못 만들었다 — 전에 쌓아 둔 예비 질문으로 채운다
        spare_question = await repo.use_spare(day, genre)
        if spare_question is None:
            raise GroupFailedError(
                f"{genre}: 새 질문도 예비 질문도 없음 "
                f"(후보 {len(candidates)}개 모두 최근 질문과 겹침)"
            )
        used.add(spare_question)
        return

    question, option_a, option_b = fresh[0]
    if await repo.add_question(
        question_date=day,
        music_genre=genre,
        question=question,
        option_a=option_a,
        option_b=option_b,
        model_version=model,
    ):
        used.add(question)

    # 나머지 후보는 버리지 않고 예비로 쌓아 둔다 — 다음에 새 질문을 못 만들 때 쓴다
    for extra_question, extra_a, extra_b in fresh[1:]:
        await repo.add_spare(
            question_date=day,
            music_genre=genre,
            question=extra_question,
            option_a=extra_a,
            option_b=extra_b,
            model_version=model,
        )
        used.add(extra_question)
