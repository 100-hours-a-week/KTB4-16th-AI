"""기능5 기억 검색 처리 순서 — 1단계(기준선): 질문 전체로 벡터 검색, 상위 N개를 템플릿으로 답한다.

  질문 → 임베딩 1번 → 내 기록에서 코멘트·곡 분위기 벡터 검색 → 합치기·최소 유사도 →
  상위 N개의 상세를 백엔드 MySQL에서 읽기 → 템플릿 답변

규칙으로 질문 나누기, 기간·날씨 조건, 장소, 사진 검색, LLM 답변은 이 위에 단계적으로 얹는다.
이 단계의 결과가 그때 "좋아졌는지" 비교하는 기준이 된다.
"""

import logging

from app.components.embedder import Embedder
from app.components.memory_answer import compose_answer
from app.components.rank_fusion import FusedRecord, fuse
from app.components.vector_search import MemoryVectorSearch
from app.db.repositories.memory_record_repository import MemoryRecord, MemoryRecordSource
from app.schemas.common import Place, Track
from app.schemas.memory_search import MatchedRecord, MemorySearchRequest, MemorySearchResponse

logger = logging.getLogger("muro.memory_search")

# 삭제됐거나 상세를 못 읽는 자물쇠가 섞여 있을 수 있어서, 돌려줄 개수보다 넉넉히 상세를 읽는다
FETCH_MULTIPLIER = 3


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
    ):
        self._embedder = embedder
        self._search = search
        self._records = records
        self._min_similarity = min_similarity
        self._comment_weight = comment_weight
        self._music_weight = music_weight
        self._candidates = candidates

    async def search(self, req: MemorySearchRequest) -> MemorySearchResponse:
        query_vector = await self._embedder.embed(req.query_text)

        # 같은 DB 세션을 쓰므로 하나씩 순서대로 실행한다
        comment_hits = await self._search.similar_by_comment(
            req.user_id, query_vector, self._candidates
        )
        music_hits = await self._search.similar_by_music(
            req.user_id, query_vector, self._candidates
        )

        ranked = fuse(
            comment_hits,
            music_hits,
            comment_weight=self._comment_weight,
            music_weight=self._music_weight,
            min_similarity=self._min_similarity,
        )
        matched = await self._load_top(req, ranked)

        # 질문 원문은 개인정보일 수 있어 남기지 않는다 — 개수만
        logger.info(
            "기억 검색 user=%s 후보(코멘트 %d, 곡 %d) → 합침 %d → 반환 %d",
            req.user_id,
            len(comment_hits),
            len(music_hits),
            len(ranked),
            len(matched),
        )
        return MemorySearchResponse(
            answer_text=compose_answer([record for record, _ in matched]),
            matched_records=[_to_matched(record, fused) for record, fused in matched],
            retry_count=0,
        )

    async def _load_top(
        self, req: MemorySearchRequest, ranked: list[FusedRecord]
    ) -> list[tuple[MemoryRecord, FusedRecord]]:
        """순위대로 상세를 읽고, 읽힌 기록만 limit개까지 남긴다.

        다른 사용자의 기록이나 삭제된 기록은 SQL 조건 때문에 안 읽혀서 건너뛴다.
        """
        wanted = ranked[: req.limit * FETCH_MULTIPLIER]
        if not wanted:
            return []
        details = await self._records.get_many(
            user_id=req.user_id, record_ids=[f.record_id for f in wanted]
        )
        matched = []
        for fused in wanted:
            record = details.get(fused.record_id)
            if record is None:
                continue
            matched.append((record, fused))
            if len(matched) == req.limit:
                break
        return matched


def _to_matched(record: MemoryRecord, fused: FusedRecord) -> MatchedRecord:
    # 백엔드가 곡을 "spotify:track:ID"나 ID로 저장할 수 있다(기능4도 같은 정규화를 함) — ID만 남긴다
    track_id = record.external_track_id.rsplit(":", 1)[-1]
    return MatchedRecord(
        record_id=str(record.record_id),
        track=Track(
            title=record.title,
            artist_name=record.artist_name,
            external_track_id=track_id,
            spotify_uri=f"spotify:track:{track_id}",
            album_image_url=record.album_image_url,
            external_url=record.external_url,
        ),
        # 1단계는 위경도를 읽지 않는다 — 법정동 이름만
        place=Place(place_id=record.place_id, name=record.place_name),
        created_at=record.created_at,
        similarity_score=round(fused.score, 4),
    )
