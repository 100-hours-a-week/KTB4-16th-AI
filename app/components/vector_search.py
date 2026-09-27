"""pgvector 유사도 검색 — 기능1(개인 히스토리 매칭)이 쓰고, 기능2·5가 이어서 재사용할 예정.

기능4가 자물쇠를 저장할 때 이미 만들어둔 벡터를 그대로 비교만 한다(CLIP·LLM 추가 호출 없음).
SQL로 "날씨=비 AND 시간대=저녁"처럼 정확히 일치하는 기록만 찾으면 "쌀쌀한 밤"과
"비 오는 저녁"처럼 말은 다르지만 느낌이 비슷한 기록을 못 찾는다 — 그래서 벡터로 찾는다.
"""

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RecordEmbedding


@dataclass(frozen=True)
class SimilarRecord:
    record_id: int
    external_track_id: str | None
    similarity: float  # 코사인 유사도 (1 - 코사인 거리), 클수록 비슷함


class RecordSearch(Protocol):
    async def similar_by_comment(
        self, user_id: int, query_vector: list[float], limit: int
    ) -> list[SimilarRecord]: ...


class VectorSearch:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def similar_by_comment(
        self, user_id: int, query_vector: list[float], limit: int
    ) -> list[SimilarRecord]:
        """이 사용자의 자물쇠 중 코멘트가 query와 가장 비슷한 것부터 limit개.

        과거 자물쇠엔 장소·날씨·시각이 저장돼 있지 않다. 그때 상황을 제일 잘 담은 게
        사용자가 직접 쓴 코멘트("비 냄새 좋았음", "혼자 걷기 좋은 밤")라서 코멘트로 찾는다.
        코멘트가 없는 자물쇠는 비교 대상에서 빠진다.
        """
        distance = RecordEmbedding.comment_embedding.cosine_distance(query_vector)
        rows = await self._session.execute(
            select(
                RecordEmbedding.record_id,
                RecordEmbedding.external_track_id,
                distance.label("distance"),
            )
            .where(
                RecordEmbedding.user_id == user_id,
                RecordEmbedding.comment_embedding.is_not(None),
            )
            .order_by(distance)
            .limit(limit)
        )
        return [
            SimilarRecord(
                record_id=row.record_id,
                external_track_id=row.external_track_id,
                similarity=1.0 - float(row.distance),
            )
            for row in rows
        ]
