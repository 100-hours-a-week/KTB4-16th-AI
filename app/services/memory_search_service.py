"""기능5 기억 검색 처리 순서.

조건이 없는 질문(곡 분위기만 묻는 질문 등)은 1단계 기준선 그대로다:
  질문 → 임베딩 1번 → 내 기록에서 코멘트·곡 분위기 벡터 검색 → 합치기·최소 유사도 →
  상위 N개의 상세를 백엔드 MySQL에서 읽기 → 템플릿 답변

질문에 조건(장소·지역·기간·계절·날씨·기온·기분·곡 무드)이 있으면 위에 이어서:
  질문 해석(규칙) → 내 자물쇠의 가벼운 정보 읽기(코멘트 포함) → 조건 점수 → 조건으로만 후보가 된
  기록의 벡터 유사도 구하기 → 벡터 점수와 조건 점수 합치기 → "모든 조건에 맞는 기록"은 전체(최대
  limit개)를 결과로, "일부만 맞는 기록"은 화면에 먼저 보이는 5건이 안 찰 때만 답변 문장의 제안
  ("이런 자물쇠를 찾으시는 건 아닌가요?")으로

답변 문장에는 자물쇠 목록을 넣지 않는다 — 프론트가 matchedRecords로 카드를 보여준다(앞 5건, 나머지는
더보기). 문장은 앞머리(질문한 조건 + 건수, LLM 또는 템플릿) + 제안 + 안내로 이뤄진다.

장소는 좌표로 비교하되, 좌표로 비교할 수 없으면 코멘트에서 그 이름을 찾는다:
  사전에 없는 장소("광교산") → 코멘트에 그 글자가 있는 기록 / 사전에 있어도 반경 안에 내 기록이
  하나도 없는 장소 → 코멘트에 이름·별칭이 있는 기록. 코멘트 어디에도 없는 사전 밖 장소는 조건에서
  빼고, 답변 끝에 "아직 모르는 장소, 동·구 이름으로 물어보세요"라고 안내한다.

사진 검색과 LLM 답변은 이 위에 단계적으로 얹는다. 조건이 없는 질문의 결과가 그때 "좋아졌는지"
비교하는 기준이 된다.
"""

import asyncio
import logging
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta, timezone

from app.components.answer_headline import HeadlineWriter
from app.components.condition_scoring import (
    ConditionWeights,
    dropped_place_words,
    kept_place_words,
    resolve_conditions,
    score_rows,
    top_candidates,
)
from app.components.embedder import Embedder
from app.components.memory_answer import (
    compose_answer,
    template_headline,
    unknown_place_notices,
)
from app.components.query_conditions import Condition
from app.components.question_parser import QuestionParser
from app.components.rank_fusion import combine_with_conditions, fuse
from app.components.vector_search import MemoryVectorSearch, SimilarRecord
from app.db.repositories.memory_record_repository import (
    ConditionRow,
    MemoryRecord,
    MemoryRecordSource,
)
from app.schemas.memory_search import (
    MemoryRecordItem,
    MemorySearchRequest,
    MemorySearchResponse,
    SuggestedRecordItem,
)

logger = logging.getLogger("muro.memory_search")

# 삭제됐거나 상세를 못 읽는 자물쇠가 섞여 있을 수 있어서, 돌려줄 개수보다 조금 더 상세를 읽는다
# (조건 검색은 SQL에서 삭제·타인 기록을 이미 거르므로 많이 읽을 필요가 없다)
FETCH_SLACK = 10

# "지난달" 같은 기간 표현의 기준이 되는 오늘은 한국 시간이다
KST = timezone(timedelta(hours=9))


def today_kst() -> date:
    return datetime.now(KST).date()


class MemorySearchService:
    def __init__(
        self,
        embedder: Embedder,
        search: MemoryVectorSearch,
        records: MemoryRecordSource,
        *,
        min_similarity: float,
        comment_weight: float,
        music_weight: float,
        candidates: int,
        parser: QuestionParser | None = None,
        condition_weight: float = 0.5,
        condition_weights: ConditionWeights | None = None,
        scan_limit: int = 5000,
        condition_candidates: int = 50,
        display_count: int = 5,
        suggestion_count: int = 3,
        headline_writer: HeadlineWriter | None = None,
        headline_timeout: float = 8.0,
        today: Callable[[], date] = today_kst,
    ):
        self._embedder = embedder
        self._search = search
        self._records = records
        self._min_similarity = min_similarity
        self._comment_weight = comment_weight
        self._music_weight = music_weight
        self._candidates = candidates
        self._parser = parser  # 없으면 조건 해석을 하지 않고 1단계 기준선으로만 동작한다
        self._condition_weight = condition_weight
        self._condition_weights = condition_weights or ConditionWeights()
        self._scan_limit = scan_limit
        self._condition_candidates = condition_candidates
        self._display_count = display_count  # 채팅에 먼저 보이는 건수 (나머지는 "전체보기" 임시 창)
        self._suggestion_count = suggestion_count  # 제안 토글을 열면 보이는 건수
        self._headline_writer = headline_writer  # 없으면 앞머리는 템플릿 문장
        self._headline_timeout = headline_timeout
        self._today = today

    async def search(self, req: MemorySearchRequest) -> MemorySearchResponse:
        conditions = self._parser.parse(req.query_text, self._today()) if self._parser else []
        query_vector = await self._embedder.embed(req.query_text)

        # 같은 DB 세션을 쓰므로 하나씩 순서대로 실행한다
        comment_hits = await self._search.similar_by_comment(
            req.user_id, query_vector, self._candidates
        )
        music_hits = await self._search.similar_by_music(
            req.user_id, query_vector, self._candidates
        )

        if conditions:
            rows = await self._records.list_condition_rows(
                user_id=req.user_id, limit=self._scan_limit
            )
            # 동·지역·장소 후보를 이 사용자의 기록으로 확인한다.
            # 걸러지고 남은 조건이 없으면 일반 검색
            asked = conditions
            conditions = resolve_conditions(asked, rows)
            notices = unknown_place_notices(
                found=kept_place_words(conditions), missing=dropped_place_words(asked, conditions)
            )
            if conditions:
                return await self._search_with_conditions(
                    req, conditions, rows, query_vector, comment_hits, music_hits, notices
                )
            return await self._search_by_similarity(req, comment_hits, music_hits, notices)
        return await self._search_by_similarity(req, comment_hits, music_hits)

    async def _search_by_similarity(
        self,
        req: MemorySearchRequest,
        comment_hits: list[SimilarRecord],
        music_hits: list[SimilarRecord],
        notices: Sequence[str] = (),
    ) -> MemorySearchResponse:
        ranked = fuse(
            comment_hits,
            music_hits,
            comment_weight=self._comment_weight,
            music_weight=self._music_weight,
            min_similarity=self._min_similarity,
        )
        wanted = ranked[: req.limit + FETCH_SLACK]
        details = (
            await self._records.get_many(
                user_id=req.user_id, record_ids=[f.record_id for f in wanted]
            )
            if wanted
            else {}
        )
        matched = _readable(wanted, details, req.limit)

        # 질문 원문은 개인정보일 수 있어 남기지 않는다 — 개수만
        logger.info(
            "기억 검색 user=%s 후보(코멘트 %d, 곡 %d) → 합침 %d → 반환 %d",
            req.user_id,
            len(comment_hits),
            len(music_hits),
            len(ranked),
            len(matched),
        )
        # 조건이 없는 질문은 앞머리에 나열할 조건이 없어서 LLM 없이 템플릿으로 쓴다
        headline = template_headline([], found=len(matched))
        return MemorySearchResponse(
            answer=compose_answer(headline, notices=notices),
            records=[_to_item(record, item.score) for record, item in matched],
        )

    async def _search_with_conditions(
        self,
        req: MemorySearchRequest,
        conditions: list[Condition],
        rows: Sequence[ConditionRow],
        query_vector: list[float],
        comment_hits: list[SimilarRecord],
        music_hits: list[SimilarRecord],
        notices: Sequence[str],
    ) -> MemorySearchResponse:
        # 곡 무드 태그는 AI DB에 있다. 곡 무드 조건이 있을 때만 읽는다
        moods = (
            await self._search.moods_by_record(req.user_id)
            if any(c.kind == "song_mood" for c in conditions)
            else {}
        )
        music_similarity = {h.record_id: h.similarity for h in music_hits}

        def score(similarity: dict[int, float]):
            return score_rows(
                rows,
                conditions,
                moods=moods,
                music_similarity=similarity,
                min_similarity=self._min_similarity,
                weights=self._condition_weights,
            )

        # 조건으로만 후보가 된 기록(위 두 검색의 상위 후보에 없는 것)은 유사도를 따로 구한다
        known_ids = {h.record_id for h in comment_hits} | set(music_similarity)
        extra_ids = top_candidates(
            score(music_similarity), exclude=known_ids, limit=self._condition_candidates
        )
        if extra_ids:
            extra_comment, extra_music = await self._search.similarities_for(
                req.user_id, extra_ids, query_vector
            )
            comment_hits = [*comment_hits, *extra_comment]
            music_hits = [*music_hits, *extra_music]
            music_similarity |= {h.record_id: h.similarity for h in extra_music}

        vector = fuse(
            comment_hits,
            music_hits,
            comment_weight=self._comment_weight,
            music_weight=self._music_weight,
            min_similarity=self._min_similarity,
        )
        ranking = combine_with_conditions(
            vector, score(music_similarity), condition_weight=self._condition_weight
        )

        # 모든 조건에 맞는 기록(전체, 최대 limit개)과 제안 후보의 상세를 한 번에 읽는다.
        # 제안은 채팅에 먼저 보이는 건수(display_count)가 안 찰 때만, 최대 suggestion_count개다.
        shown = min(req.limit, self._display_count)
        exact = ranking.exact[: req.limit + FETCH_SLACK]
        suggested = ranking.suggestions[: self._suggestion_count + FETCH_SLACK]
        wanted_ids = [r.record_id for r in (*exact, *suggested)]
        details = (
            await self._records.get_many(user_id=req.user_id, record_ids=wanted_ids)
            if wanted_ids
            else {}
        )
        matched = _readable(exact, details, req.limit)
        suggestions = (
            _readable(suggested, details, self._suggestion_count) if len(matched) < shown else []
        )

        logger.info(
            "기억 검색(조건 %s) user=%s 자물쇠 %d → 모두 맞음 %d, 일부 맞음 %d → 반환 %d, 제안 %d",
            ",".join(c.kind for c in conditions),
            req.user_id,
            len(rows),
            len(ranking.exact),
            len(ranking.suggestions),
            len(matched),
            len(suggestions),
        )
        headline = await self._headline(
            [c.label for c in conditions],
            found=len(matched),
            total=len(ranking.exact),
            suggestions=len(suggestions),
        )
        return MemorySearchResponse(
            answer=compose_answer(headline, notices=notices),
            records=[_to_item(record, item.score) for record, item in matched],
            suggested_records=[
                _to_suggested_item(record, item.score, item.condition.matched)
                for record, item in suggestions
            ],
        )

    async def _headline(
        self, labels: Sequence[str], *, found: int, total: int, suggestions: int
    ) -> str:
        """앞머리 문장. LLM이 쓰되 못 쓰면(꺼짐·실패·시간 초과·검증 실패) 템플릿으로 대신한다."""
        fallback = template_headline(labels, found=found, total=total, suggestions=suggestions)
        if self._headline_writer is None or (found == 0 and suggestions == 0):
            return fallback  # 결과가 없으면 근거가 없어서 LLM을 부르지 않는다
        try:
            text = await asyncio.wait_for(
                self._headline_writer.write(
                    labels, found=found, total=total, suggestions=suggestions
                ),
                timeout=self._headline_timeout,
            )
        except Exception as exc:  # 앞머리는 꾸밈이라 실패해도 검색 결과는 그대로 돌려준다
            logger.warning("기억 검색 앞머리 LLM 실패(%s) — 템플릿으로 대신", type(exc).__name__)
            return fallback
        if text is None:
            logger.warning("기억 검색 앞머리 LLM 응답이 기준에 안 맞아 템플릿 문장으로 대신")
            return fallback
        return text


def _readable(ranked: Sequence, details: dict[int, MemoryRecord], limit: int) -> list[tuple]:
    """순위대로 상세를 읽을 수 있는 기록만 limit개까지 남긴다.

    다른 사용자의 기록이나 삭제된 기록은 SQL 조건 때문에 안 읽혀서 건너뛴다.
    """
    kept = []
    for item in ranked:
        record = details.get(item.record_id)
        if record is None:
            continue
        kept.append((record, item))
        if len(kept) == limit:
            break
    return kept


def _to_item(record: MemoryRecord, score: float) -> MemoryRecordItem:
    # 백엔드가 곡을 "spotify:track:ID"나 ID로 저장할 수 있다(기능4도 같은 정규화를 함) — ID만 남긴다
    track_id = record.external_track_id.rsplit(":", 1)[-1]
    d = record.created_at  # 백엔드 DB는 한국시간으로 저장한다
    return MemoryRecordItem(
        record_id=record.record_id,
        music_track_id=record.music_track_id,
        title=record.title,
        artist_name=record.artist_name,
        place_id=record.place_id,
        # 위경도는 읽지 않는다 — 법정동 이름만
        place_name=record.place_name,
        date=f"{d.year}.{d.month:02d}.{d.day:02d}",
        weather_condition=record.weather_condition,
        external_track_id=track_id,
        album_image_url=record.album_image_url,
        external_url=record.external_url,
        similarity_score=round(score, 4),
    )


def _to_suggested_item(
    record: MemoryRecord, score: float, matched_conditions: Sequence[str]
) -> SuggestedRecordItem:
    return SuggestedRecordItem(
        **_to_item(record, score).model_dump(), matched_conditions=list(matched_conditions)
    )
