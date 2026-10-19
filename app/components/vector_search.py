"""pgvector 유사도 검색 — 기능1(개인 히스토리 매칭)이 쓰고, 기능2·5가 이어서 재사용할 예정.

기능4가 자물쇠를 저장할 때 이미 만들어둔 벡터를 그대로 비교만 한다(CLIP·LLM 추가 호출 없음).
SQL로 "날씨=비 AND 시간대=저녁"처럼 정확히 일치하는 기록만 찾으면 "쌀쌀한 밤"과
"비 오는 저녁"처럼 말은 다르지만 느낌이 비슷한 기록을 못 찾는다 — 그래서 벡터로 찾는다.
"""

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import RecordEmbedding, TrackMood


@dataclass(frozen=True)
class SimilarRecord:
    record_id: int
    external_track_id: str | None
    similarity: float  # 코사인 유사도 (1 - 코사인 거리), 클수록 비슷함


class RecordSearch(Protocol):
    async def similar_by_comment(
        self, user_id: int, query_vector: list[float], limit: int
    ) -> list[SimilarRecord]: ...

    async def music_vectors(self, user_id: int, limit: int) -> list[list[float]]: ...

    async def music_vectors_for_records(
        self, user_id: int, record_ids: list[int]
    ) -> list[list[float]]: ...


class MemoryVectorSearch(Protocol):
    """기능5 기억 검색이 쓰는 두 검색 — 코멘트 벡터와 곡 분위기 벡터."""

    async def similar_by_comment(
        self, user_id: int, query_vector: list[float], limit: int
    ) -> list[SimilarRecord]: ...

    async def similar_by_music(
        self, user_id: int, query_vector: list[float], limit: int
    ) -> list[SimilarRecord]: ...

    async def similarities_for(
        self, user_id: int, record_ids: list[int], query_vector: list[float]
    ) -> tuple[list[SimilarRecord], list[SimilarRecord]]: ...

    async def moods_by_record(self, user_id: int) -> dict[int, tuple[str, ...]]: ...


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

    async def music_vectors(self, user_id: int, limit: int) -> list[list[float]]:
        """이 사용자가 저장한 곡들의 분위기 벡터, 최근 것부터 limit개 (취향 벡터의 재료)."""
        rows = await self._session.execute(
            select(RecordEmbedding.music_embedding)
            .where(
                RecordEmbedding.user_id == user_id,
                RecordEmbedding.music_embedding.is_not(None),
            )
            .order_by(RecordEmbedding.created_at.desc())
            .limit(limit)
        )
        return [list(row.music_embedding) for row in rows]

    async def music_vectors_for_records(
        self, user_id: int, record_ids: list[int]
    ) -> list[list[float]]:
        """이 사용자의 자물쇠 중 record_ids에 해당하는 곡 분위기 벡터 (상황이 맞는 기록의 취향)."""
        if not record_ids:
            return []
        rows = await self._session.execute(
            select(RecordEmbedding.music_embedding).where(
                RecordEmbedding.user_id == user_id,
                RecordEmbedding.record_id.in_(record_ids),
                RecordEmbedding.music_embedding.is_not(None),
            )
        )
        return [list(row.music_embedding) for row in rows]

    async def similar_by_music(
        self, user_id: int, query_vector: list[float], limit: int
    ) -> list[SimilarRecord]:
        """이 사용자의 자물쇠 중 곡 분위기 설명이 query와 가장 비슷한 것부터 limit개.

        record_embeddings.music_embedding은 LLM이 곡 제목·아티스트로 쓴 무드 설명을
        임베딩한 값이다(기능4). 코멘트를 안 쓴 자물쇠도 항상 이 벡터가 있다.
        """
        distance = RecordEmbedding.music_embedding.cosine_distance(query_vector)
        rows = await self._session.execute(
            select(
                RecordEmbedding.record_id,
                RecordEmbedding.external_track_id,
                distance.label("distance"),
            )
            .where(
                RecordEmbedding.user_id == user_id,
                RecordEmbedding.music_embedding.is_not(None),
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

    async def similarities_for(
        self, user_id: int, record_ids: list[int], query_vector: list[float]
    ) -> tuple[list[SimilarRecord], list[SimilarRecord]]:
        """지정한 자물쇠들의 질문과의 유사도를 (코멘트, 곡 분위기) 순서로 돌려준다.

        조건(장소·계절 등)만으로 후보가 된 기록은 위 두 검색의 상위 후보에 없어서 유사도가 없다.
        순위를 정하려고 그 기록들만 따로 구한다. 다른 사용자의 기록은 user_id 조건으로 빠진다.
        """
        if not record_ids:
            return [], []

        comment_distance = RecordEmbedding.comment_embedding.cosine_distance(query_vector)
        comment_rows = await self._session.execute(
            select(
                RecordEmbedding.record_id,
                RecordEmbedding.external_track_id,
                comment_distance.label("distance"),
            ).where(
                RecordEmbedding.user_id == user_id,
                RecordEmbedding.record_id.in_(record_ids),
                RecordEmbedding.comment_embedding.is_not(None),
            )
        )
        music_distance = RecordEmbedding.music_embedding.cosine_distance(query_vector)
        music_rows = await self._session.execute(
            select(
                RecordEmbedding.record_id,
                RecordEmbedding.external_track_id,
                music_distance.label("distance"),
            ).where(
                RecordEmbedding.user_id == user_id,
                RecordEmbedding.record_id.in_(record_ids),
                RecordEmbedding.music_embedding.is_not(None),
            )
        )
        return _to_hits(comment_rows), _to_hits(music_rows)

    async def moods_by_record(self, user_id: int) -> dict[int, tuple[str, ...]]:
        """이 사용자의 자물쇠마다 곡 무드 태그(track_moods.moods)를 돌려준다.

        태그가 아직 없는 곡(예전에 만든 캐시)은 빈 튜플이다. 곡 정보가 없는 자물쇠는 빠진다.
        """
        rows = await self._session.execute(
            select(RecordEmbedding.record_id, TrackMood.moods)
            .join(TrackMood, RecordEmbedding.external_track_id == TrackMood.external_track_id)
            .where(RecordEmbedding.user_id == user_id)
        )
        return {row.record_id: tuple(row.moods or ()) for row in rows}


def _to_hits(rows) -> list[SimilarRecord]:
    return [
        SimilarRecord(
            record_id=row.record_id,
            external_track_id=row.external_track_id,
            similarity=1.0 - float(row.distance),
        )
        for row in rows
    ]
