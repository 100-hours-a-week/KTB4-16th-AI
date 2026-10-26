"""기능5 기억 검색 처리 순서.

조건이 없는 질문(곡 분위기만 묻는 질문 등)은 1단계 기준선 그대로다:
  질문 → 임베딩 1번 → 내 기록에서 코멘트·곡 분위기 벡터 검색 → 합치기·최소 유사도 →
  상위 N개가 이 사용자의 삭제되지 않은 자물쇠인지 백엔드 MySQL에서 확인 → 템플릿 답변

질문에 조건(장소·지역·기간·계절·날씨·요일·시간대·기온·기분·장르·곡 무드, 그리고 이 사용자가
저장한 곡의 가수·곡명)이 있으면 위에 이어서:
  질문 해석(규칙) → 내 자물쇠의 가벼운 정보 읽기(코멘트 포함) → 조건 점수 → 조건으로만 후보가 된
  기록의 벡터 유사도 구하기 → 벡터 점수와 조건 점수 합치기 → "모든 조건에 맞는 기록"은 전체(최대
  limit개)를 결과로, "일부만 맞는 기록"은 화면에 먼저 보이는 5건이 안 찰 때만 답변 문장의 제안
  ("이런 자물쇠를 찾으시는 건 아닌가요?")으로

응답은 ID만 준다: records는 recordId만, suggestedRecords는 recordId와 matchedConditions만.
카드 정보는 백엔드가 recordId로 자기 DB에서 만든다. 답변 문장에는 자물쇠 목록도, 건수 숫자도
넣지 않는다 — 건수는 프론트가 records.length로 보여준다. 문장은 앞머리(질문한 조건 + 결과 유무,
LLM 또는 템플릿) + 안내로 이뤄진다.

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
    corrected_name_notices,
    template_headline,
    unknown_place_notices,
)
from app.components.query_conditions import Condition
from app.components.question_parser import QuestionParser
from app.components.rank_fusion import combine_with_conditions, fuse
from app.components.track_lexicon import TrackLexicon, artist_name_keys
from app.components.vector_search import MemoryVectorSearch, SimilarRecord
from app.db.repositories.artist_alias_repository import ArtistAliasSource
from app.db.repositories.memory_record_repository import ConditionRow, MemoryRecordSource
from app.exceptions import UpstreamError
from app.schemas.memory_search import (
    MemoryRecordItem,
    MemorySearchRequest,
    MemorySearchResponse,
    SuggestedRecordItem,
)

logger = logging.getLogger("muro.memory_search")

# 삭제됐거나 다른 사용자의 자물쇠가 섞여 있을 수 있어서, 돌려줄 개수보다 조금 더 확인한다
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
        aliases: ArtistAliasSource | None = None,
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
        self._aliases = aliases  # 없으면 저장된 이름 그대로만 가수를 찾는다

    async def search(self, req: MemorySearchRequest) -> MemorySearchResponse:
        conditions = (
            self._parser.parse(req.query_text, self._today(), await self._lexicon(req.user_id))
            if self._parser
            else []
        )
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
            notices = [
                *corrected_name_notices(conditions),
                *unknown_place_notices(
                    found=kept_place_words(conditions),
                    missing=dropped_place_words(asked, conditions),
                ),
            ]
            if conditions:
                return await self._search_with_conditions(
                    req, conditions, rows, query_vector, comment_hits, music_hits, notices
                )
            return await self._search_by_similarity(req, comment_hits, music_hits, notices)
        return await self._search_by_similarity(req, comment_hits, music_hits)

    async def _lexicon(self, user_id: int) -> TrackLexicon:
        """이 사용자가 저장한 곡의 가수·곡명 사전. 질문에서 가수·곡명을 알아보는 데 쓴다."""
        try:
            names = await self._records.list_track_names(user_id=user_id, limit=self._scan_limit)
        except UpstreamError:
            # 이름 목록은 검색을 돕는 부가 정보다. 못 읽어도 가수·곡명 없이 평소처럼 검색한다
            logger.warning("기억 검색 곡 목록 읽기 실패 — 가수·곡명 조건 없이 진행", exc_info=True)
            return TrackLexicon.build([])
        return TrackLexicon.build(names, await self._artist_aliases(names))

    async def _artist_aliases(self, names) -> dict[str, list[str]]:
        """저장된 가수 이름의 다른 표기(영어↔한글). 못 읽어도 저장된 이름만으로 평소처럼 찾는다."""
        if self._aliases is None:
            return {}
        try:
            return await self._aliases.aliases_for(artist_name_keys(names))
        except UpstreamError:
            logger.warning("가수 별칭 읽기 실패 — 별칭 없이 진행", exc_info=True)
            return {}

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
        existing = (
            await self._records.existing_ids(
                user_id=req.user_id, record_ids=[f.record_id for f in wanted]
            )
            if wanted
            else set()
        )
        matched = _owned(wanted, existing, req.limit)

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
            records=[MemoryRecordItem(record_id=item.record_id) for item in matched],
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
        # 곡 장르 태그는 AI DB에 있다. 장르 조건이 있을 때만 읽는다
        genres = (
            await self._search.genres_by_record(req.user_id)
            if any(c.kind == "genre" for c in conditions)
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
                genres=genres,
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
        existing = (
            await self._records.existing_ids(user_id=req.user_id, record_ids=wanted_ids)
            if wanted_ids
            else set()
        )
        matched = _owned(exact, existing, req.limit)
        suggestions = (
            _owned(suggested, existing, self._suggestion_count) if len(matched) < shown else []
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
            llm_labels=[_llm_label(c) for c in conditions],
        )
        return MemorySearchResponse(
            answer=compose_answer(headline, notices=notices),
            records=[MemoryRecordItem(record_id=item.record_id) for item in matched],
            suggested_records=[
                SuggestedRecordItem(
                    record_id=item.record_id, matched_conditions=list(item.condition.matched)
                )
                for item in suggestions
            ],
        )

    async def _headline(
        self,
        labels: Sequence[str],
        *,
        found: int,
        total: int,
        suggestions: int,
        llm_labels: Sequence[str] | None = None,
    ) -> str:
        """앞머리 문장. LLM이 쓰되 못 쓰면(꺼짐·실패·시간 초과·검증 실패) 템플릿으로 대신한다.

        labels는 템플릿 문장(코드)에 쓰는 조건 이름이고, llm_labels는 LLM에 보내는 조건 이름이다.
        가수·곡명은 사용자의 음악 정보라서 LLM에는 이름 대신 "가수"·"곡명"만 보낸다.
        """
        fallback = template_headline(labels, found=found, total=total, suggestions=suggestions)
        if self._headline_writer is None or (found == 0 and suggestions == 0):
            return fallback  # 결과가 없으면 근거가 없어서 LLM을 부르지 않는다
        try:
            text = await asyncio.wait_for(
                self._headline_writer.write(
                    llm_labels if llm_labels is not None else labels,
                    found=found,
                    total=total,
                    suggestions=suggestions,
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


def _llm_label(cond: Condition) -> str:
    """LLM에 보내는 조건 이름. 가수·곡명은 이름을 숨기고 종류만 보낸다."""
    return {"artist": "가수", "title": "곡명"}.get(cond.kind, cond.label)


def _owned(ranked: Sequence, existing: set[int], limit: int) -> list:
    """순위대로 이 사용자의 삭제되지 않은 자물쇠만 limit개까지 남긴다.

    다른 사용자의 기록이나 삭제된 기록은 SQL 조건 때문에 existing에 없어서 건너뛴다.
    """
    kept = []
    for item in ranked:
        if item.record_id not in existing:
            continue
        kept.append(item)
        if len(kept) == limit:
            break
    return kept
