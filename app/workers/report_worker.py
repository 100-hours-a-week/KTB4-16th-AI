"""RECAP 배치 작업 실행 단위."""

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.components.report_summarizer import ReportSummarizer
from app.db.mysql import get_sessionmaker as get_mysql_sessionmaker
from app.db.postgres import get_sessionmaker as get_postgres_sessionmaker
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.report_repository import RecordSummary, ReportRepository
from app.dependencies import Clients, get_clients

JOB_TYPE = "report"


@dataclass(frozen=True)
class MonthlyReportResult:
    """이번 달 RECAP 결과 묶음. 저장 위치(AI 소유 테이블? 백엔드 콜백?)는 아직 미정 —
    지금은 이 형태로 로그에만 남긴다."""

    ai_recap: str
    top_artist: str | None
    top_place: str | None
    avg_mood: float | None
    photo_category_distribution: dict[str, float]


# "이 배치(year-month) 알림을 이미 보냈는지"를 ai_jobs의 기존 유니크 제약으로
# 체크하기 위한 전용 job_type. 실제 처리 작업이 아니라 "알림 1회 전송권" 표식일 뿐이다.
BATCH_NOTIFY_JOB_TYPE = "report_batch_notify"

logger = logging.getLogger("muro.worker.report")

# 백엔드 records.weather_condition ENUM(Flyway V1) 7종, 기상청 단기예보 기준 표현.
# CLOUDY는 "구름 많음", OVERCAST가 "흐림"이다. 모르는 값은 원본 그대로 쓴다.
WEATHER_KO = {
    "CLEAR": "맑음",
    "CLOUDY": "구름 많음",
    "OVERCAST": "흐림",
    "RAIN": "비",
    "SNOW": "눈",
    "RAIN_SNOW": "진눈깨비",
    "SHOWER": "소나기",
}


def _describe_weather(raw: str) -> str:
    return WEATHER_KO.get(raw.upper(), raw)


async def handle(payload: dict[str, Any]) -> None:
    user_id: int = payload["user_id"]
    year: int = payload["year"]
    month: int = payload["month"]

    async with (
        get_mysql_sessionmaker()() as mysql_session,
        get_postgres_sessionmaker()() as pg_session,
    ):
        repo = ReportRepository(mysql_session, pg_session)
        records = await repo.get_records(user_id=user_id, year=year, month=month)
        record_ids = [r.record_id for r in records]
        moods = await repo.get_mood_texts(record_ids)
        image_vectors = await repo.get_image_embeddings(record_ids)
        stats = await repo.get_stats(user_id=user_id, year=year, month=month)

    photo_tags_by_record = await _classify_photos(image_vectors)
    photo_category_distribution = _category_distribution(photo_tags_by_record)
    record_lines = _build_record_lines(records, moods, photo_tags_by_record)

    clients = get_clients()
    summarizer = ReportSummarizer(clients.llm_general)
    summary = await summarizer.summarize(record_lines, stats, photo_category_distribution)

    result = MonthlyReportResult(
        ai_recap=summary,
        top_artist=stats.top_artist,
        top_place=stats.top_place,
        avg_mood=stats.avg_mood_score,
        photo_category_distribution=photo_category_distribution,
    )
    # TODO: result를 실제로 어디에 저장/전달할지는 아직 미정
    # (AI 소유 monthly_reports 테이블? 백엔드가 가져갈 조회 API?) —
    # 확정 전이라 지금은 로그로만 남긴다. 콜백에는 이 내용이 안 실린다(AI API 시트 기준).
    logger.info("report 완료 user=%s %s-%s: %s", user_id, year, month, result)

    await _notify_if_batch_complete(clients, year=year, month=month, user_id=user_id)


async def _notify_if_batch_complete(
    clients: Clients, *, year: int, month: int, user_id: int
) -> None:
    """이 job으로 배치가 다 끝났으면 백엔드에 딱 한 번 알린다.

    1. 내 job을 먼저 done으로 표시한다 — worker_main의 mark_done을 기다리면
       "남은 개수 세기"가 나 자신을 아직 미완료로 착각한다.
    2. 남은(pending·running) job이 0개면 배치가 끝난 것.
    3. 그래도 여러 job이 동시에 끝나 동시에 0개를 볼 수 있으니, 유니크 제약으로
       "알림 전송권"을 한 번만 발급해서 실제로는 그중 하나만 진짜로 보낸다.
    """
    dedupe_key = f"{year}-{month}-{user_id}"
    dedupe_prefix = f"{year}-{month}-"

    async with get_postgres_sessionmaker()() as session:
        jobs = JobRepository(session)
        await jobs.mark_done_by_dedupe_key(job_type=JOB_TYPE, dedupe_key=dedupe_key)
        remaining = await jobs.count_incomplete(job_type=JOB_TYPE, dedupe_key_prefix=dedupe_prefix)
        if remaining > 0:
            return  # 아직 다른 사용자 job이 처리 중 — 마지막 job이 알림을 보낼 것

        claimed = await jobs.try_claim_once(
            job_type=BATCH_NOTIFY_JOB_TYPE, dedupe_key=f"{year}-{month}"
        )
        if not claimed:
            return  # 다른 job이 이미 이 배치의 알림을 보냈음(동시 완료 레이스)

        user_ids = await jobs.completed_user_ids(job_type=JOB_TYPE, dedupe_key_prefix=dedupe_prefix)

    await clients.backend.notify_report_ready(
        year=year,
        month=month,
        user_ids=user_ids,
        generated_at=datetime.now(UTC).isoformat(),
    )
    logger.info("배치 완료 알림 전송 %s-%s: 대상 %d명", year, month, len(user_ids))


def _build_record_lines(
    records: list[RecordSummary],
    moods: dict[int, str],
    photo_tags_by_record: dict[int, list[str]],
) -> list[str]:
    """자물쇠 1건 = 요약 1줄. 곡 무드·기분점수·날씨·사진·코멘트를 한 줄로 합친다.

    값이 없는 항목(코멘트 미작성, 아직 임베딩 안 됨 등)은 그 항목만 건너뛴다 —
    한 필드가 없다고 그 자물쇠 전체를 요약에서 빼지 않는다.
    """
    lines = []
    for r in records:
        parts = []
        if mood_text := moods.get(r.record_id):
            parts.append(f"곡 분위기: {mood_text}")
        if r.artist_name:
            parts.append(f"아티스트: {r.artist_name}")
        if r.mood_score is not None:
            parts.append(f"기분점수: {r.mood_score}")  # -50(안 좋음) ~ 50(좋음)
        if r.weather_condition:
            weather = _describe_weather(r.weather_condition)
            if r.temperature is not None:
                weather += f" {r.temperature:g}도"
            parts.append(f"날씨: {weather}")
        if r.place_name:
            parts.append(f"장소: {r.place_name}")
        if tags := photo_tags_by_record.get(r.record_id):
            parts.append(f"사진: {', '.join(tags)}")
        if r.comment:
            parts.append(f"코멘트: {r.comment}")
        if parts:
            lines.append(" / ".join(parts))
    return lines


OTHER_CATEGORY = "기타"  # 카테고리 목록 자체가 비어있는 등, 정말 예외적인 경우의 방어용


async def _classify_photos(image_vectors: dict[int, list[float]]) -> dict[int, list[str]]:
    """TODO: 기능3(CLIP 제로샷 사진 태깅)이 develop에 merge된 후 실제 구현으로 교체 예정.

    현재 develop엔 ClipTagger.tag_from_vector()/get_clip_tagger()가 없어(기능3 미병합)
    임시로 빈 결과를 반환한다. 실제 구현은 로컬 stash("recap 전체버전")에 보관되어 있음
    — 기능3 병합 후 이 함수만 복원하면 된다.
    """
    return {}


def _category_distribution(photo_tags_by_record: dict[int, list[str]]) -> dict[str, float]:
    """월간 집계용 — 레코드마다 최상위 태그 1개를 모아 비중을 낸다.

    분모는 항상 "이미지 벡터가 있던 레코드 수" 전체다.
    """
    if not photo_tags_by_record:
        return {}
    counts = Counter(tags[0] if tags else OTHER_CATEGORY for tags in photo_tags_by_record.values())
    total = sum(counts.values())
    return {category: count / total for category, count in counts.items()}
