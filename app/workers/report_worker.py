"""RECAP 배치 작업 실행 단위."""

import logging
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.components.report_summarizer import ReportSummarizer
from app.config import get_settings
from app.db.mysql import get_sessionmaker as get_mysql_sessionmaker
from app.db.postgres import get_sessionmaker as get_postgres_sessionmaker
from app.db.repositories.job_repository import JobRepository
from app.db.repositories.report_repository import (
    MonthlyReportRepository,
    RecordSummary,
    ReportRepository,
)
from app.dependencies import get_clients, get_clip_tagger

JOB_TYPE = "report"


# "이 배치(year-month) 알림을 이미 보냈는지"를 ai_jobs의 기존 유니크 제약으로
# 체크하기 위한 전용 job_type. 실제 처리 작업이 아니라 "알림 1회 전송권" 표식일 뿐이다.
BATCH_NOTIFY_JOB_TYPE = "report_batch_notify"

# 백엔드: errorCode 값 자체는 정해진 목록 없음, COMPLETED/FAILED 구분만 필요하다고 확인됨.
FAILED_ERROR_CODE = "GENERATION_FAILED"

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


@dataclass(frozen=True)
class BatchScope:
    """배치 하나를 가리키는 값 — job 키 접두사와 알림 키를 한 곳에서 만든다.

    batch_request_id가 있으면 그 값으로 구분한다(같은 달 수동 트리거를 2번 보내도 서로
    다른 배치). 없으면(구버전 백엔드) 예전처럼 연·월로 구분한다.

    - job 키: "{id}:{userId}" / "{year}-{month}-{userId}". ID에는 ':'를 못 쓰게 검증해서
      "abc"의 접두사가 "abc-1"의 job과 겹치지 않는다.
    - 알림 키: "batch:{id}" / "{year}-{month}". 접두사를 붙여서 ID가 우연히 "2026-9" 같은
      모양이어도 연·월 키와 겹치지 않는다.
    """

    year: int
    month: int
    batch_request_id: str | None = None

    @property
    def job_prefix(self) -> str:
        if self.batch_request_id:
            return f"{self.batch_request_id}:"
        return f"{self.year}-{self.month}-"

    @property
    def notify_key(self) -> str:
        if self.batch_request_id:
            return f"batch:{self.batch_request_id}"
        return f"{self.year}-{self.month}"

    def job_key(self, user_id: int) -> str:
        return f"{self.job_prefix}{user_id}"

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BatchScope":
        # 이 기능이 들어오기 전에 큐에 들어간 job엔 batch_request_id가 없다 → 연·월로 동작
        return cls(
            year=payload["year"],
            month=payload["month"],
            batch_request_id=payload.get("batch_request_id"),
        )


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

    # stats(기분·아티스트·장소)는 저장 안 함 — 백엔드가 자기 MySQL 원본으로 직접 계산하기로
    # 협의됨. 위에서 요약 텍스트를 쓸 때만 내부적으로 썼다.
    scene_stats = _photo_scene_stats(photo_tags_by_record)
    async with get_postgres_sessionmaker()() as pg_session:
        await MonthlyReportRepository(pg_session).save(
            user_id=user_id,
            year=year,
            month=month,
            ai_recap_text=summary,
            photo_scenes=[_scene_stat_dict(s) for s in scene_stats],
        )
    logger.info("report 저장 완료 user=%s %s-%s", user_id, year, month)


async def on_finished(payload: dict[str, Any]) -> None:
    """worker_main이 이 job을 done 또는 failed로 확정한 뒤 부른다.

    완료 알림을 handle() 끝(성공 경로)에만 두면, 배치에서 마지막으로 끝난 job이
    실패(재시도 소진)일 때 아무도 알림을 안 보낸다 — 앞서 성공한 job들은 그때
    "아직 남은 job 있음"을 보고 넘어갔기 때문. 그래서 성공·실패 양쪽이 지나는 여기로 뺐다.
    """
    await _enqueue_notify_if_batch_complete(scope=BatchScope.from_payload(payload))


async def _enqueue_notify_if_batch_complete(*, scope: BatchScope) -> None:
    """이 job으로 배치가 다 끝났으면, 완료 알림 전송을 별도 job으로 큐에 넣는다.

    1. 남은(pending·running) job이 0개면 배치가 끝난 것. 이 job의 상태는
       worker_main이 이미 done/failed로 확정해뒀으므로 자기 자신은 안 세어진다.
    2. 그래도 여러 job이 동시에 끝나 동시에 0개를 볼 수 있으니, 유니크 제약으로
       "알림 전송권"을 한 번만 발급해서 실제로는 그중 하나만 진짜로 큐에 넣는다.
    3. 알림 전송 자체는 여기서 바로 하지 않고 report_batch_notify job으로 큐에 넣는다 —
       그래야 전송이 실패해도(백엔드 일시 장애 등) 일반 job과 똑같이 재시도된다.
       claimed 마커만 있고 바로 clients.backend.notify_report_ready를 부르면,
       그 호출 자체가 실패했을 때 재시도할 방법이 없어 배치 알림이 영영 유실된다.
    """
    async with get_postgres_sessionmaker()() as session:
        jobs = JobRepository(session)
        remaining = await jobs.count_incomplete(
            job_type=JOB_TYPE, dedupe_key_prefix=scope.job_prefix
        )
        if remaining > 0:
            return  # 아직 다른 사용자 job이 처리 중 — 마지막 job이 알림 job을 큐에 넣을 것

        claimed = await jobs.try_claim_once(
            job_type=BATCH_NOTIFY_JOB_TYPE, dedupe_key=scope.notify_key
        )
        if not claimed:
            return  # 다른 job이 이미 이 배치의 알림 job을 큐에 넣었음(동시 완료 레이스)

        await jobs.enqueue(
            job_type=BATCH_NOTIFY_JOB_TYPE,
            dedupe_key=scope.notify_key,
            payload={
                "year": scope.year,
                "month": scope.month,
                "batch_request_id": scope.batch_request_id,
            },
            max_attempts=get_settings().job_max_attempts,
        )


async def handle_batch_notify(payload: dict[str, Any]) -> None:
    """배치 완료 알림을 실제로 전송한다. 콜백엔 각 유저의 실제 리포트 내용
    (aiRecap, photoScenes)도 실어서 보낸다 — 백엔드 DTO(MonthlyReportAiCallbackRequest)
    협의 결과. 재시도를 다 써서 포기(failed)한 유저도 results에 포함한다 — status만
    FAILED로 다르고 errorCode를 채운다(백엔드: 코드 값 자체는 안 정해짐, 구분만 필요).

    이 함수가 실패(예외)하면 worker_main이 일반 job과 똑같이 재시도한다 — 전송
    실패가 조용히 유실되지 않는다.
    """
    scope = BatchScope.from_payload(payload)
    year, month = scope.year, scope.month

    async with get_postgres_sessionmaker()() as session:
        jobs = JobRepository(session)
        completed_ids = await jobs.completed_user_ids(
            job_type=JOB_TYPE, dedupe_key_prefix=scope.job_prefix
        )
        failed_ids = await jobs.failed_user_ids(
            job_type=JOB_TYPE, dedupe_key_prefix=scope.job_prefix
        )
        reports = await MonthlyReportRepository(session).get_many(
            user_ids=completed_ids, year=year, month=month
        )

    results = [
        {
            "userId": uid,
            "status": "COMPLETED",
            "aiRecap": {"text": record.ai_recap_text},
            "photoScenes": record.photo_scenes,
            "errorCode": None,
        }
        for uid, record in reports.items()
    ] + [
        {
            "userId": uid,
            "status": "FAILED",
            "aiRecap": {"text": None},
            "photoScenes": [],
            "errorCode": FAILED_ERROR_CODE,
        }
        for uid in failed_ids
    ]

    await get_clients().backend.notify_report_ready(
        job_id=f"report_batch_{year}-{month}",
        year=year,
        month=month,
        generated_at=datetime.now(UTC).isoformat(),
        results=results,
        batch_request_id=scope.batch_request_id,
    )
    logger.info(
        "배치 완료 알림 전송 %s-%s (batchRequestId=%s): 성공 %d명, 실패 %d명",
        year,
        month,
        scope.batch_request_id,
        len(reports),
        len(failed_ids),
    )


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
    """레코드별 사진 카테고리 태그(최대 1개). CLIP을 다시 호출하지 않는다.

    ClipTagger.tag_from_vector()는 이미 계산된 벡터와 캐싱된 카테고리 라벨
    벡터의 코사인 유사도만 비교한다. develop 병합본엔 임계치를 끄는 옵션이
    없어져서(항상 THRESHOLD 적용), 임계치 미달이면 빈 리스트가 올 수 있다 —
    그 경우 _category_distribution이 OTHER_CATEGORY("기타")로 처리한다.
    """
    if not image_vectors:
        return {}

    tagger = get_clip_tagger()
    result: dict[int, list[str]] = {}
    for record_id, vector in image_vectors.items():
        result[record_id] = await tagger.tag_from_vector(vector, top_k=1)
    return result


def _category_distribution(photo_tags_by_record: dict[int, list[str]]) -> dict[str, float]:
    """월간 집계용 — 레코드마다 최상위 태그 1개를 모아 비중을 낸다.

    분모는 항상 "이미지 벡터가 있던 레코드 수" 전체다.
    """
    if not photo_tags_by_record:
        return {}
    counts = Counter(tags[0] if tags else OTHER_CATEGORY for tags in photo_tags_by_record.values())
    total = sum(counts.values())
    return {category: count / total for category, count in counts.items()}


@dataclass(frozen=True)
class PhotoSceneStat:
    tag: str
    count: int
    ratio: int  # 백분율, 반올림


def _photo_scene_stats(photo_tags_by_record: dict[int, list[str]]) -> list[PhotoSceneStat]:
    """백엔드 photoScenes — 카테고리별 개수·비중(%)을 비중 높은 순으로 정리한다."""
    if not photo_tags_by_record:
        return []
    counts = Counter(tags[0] if tags else OTHER_CATEGORY for tags in photo_tags_by_record.values())
    total = sum(counts.values())
    return [
        PhotoSceneStat(tag=tag, count=count, ratio=round(count / total * 100))
        for tag, count in counts.most_common()
    ]


def _scene_stat_dict(stat: PhotoSceneStat) -> dict[str, Any]:
    return {"tag": stat.tag, "count": stat.count, "ratio": stat.ratio}
