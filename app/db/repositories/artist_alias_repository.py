"""artist_aliases 접근 — 가수 이름의 다른 표기(영어↔한글).

읽기: 기억 검색이 질문에서 가수를 찾을 때, 이 사용자의 가수들에 달린 다른 표기를 가져온다.
쓰기·원본: 워커가 record_facts의 가수 이름 중 아직 변환하지 않은 것만 골라 변환해 저장한다.
"""

from collections.abc import Collection, Sequence
from typing import Protocol

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ArtistAlias, RecordFact


class ArtistAliasSource(Protocol):
    async def aliases_for(self, name_keys: Collection[str]) -> dict[str, list[str]]: ...


class ArtistAliasRepository:
    def __init__(self, session: AsyncSession):
        self._session = session

    async def aliases_for(self, name_keys: Collection[str]) -> dict[str, list[str]]:
        """name_key → 다른 표기들. 별칭이 없는(빈 목록) 가수는 돌려주지 않는다.

        만든 버전이 옛것이어도 쓴다 — 버전이 바뀌면 워커가 새로 만들어 덮어쓴다."""
        if not name_keys:
            return {}
        rows = await self._session.execute(
            select(ArtistAlias.name_key, ArtistAlias.aliases).where(
                ArtistAlias.name_key.in_(list(name_keys)),
                func.cardinality(ArtistAlias.aliases) > 0,
            )
        )
        return {row.name_key: list(row.aliases) for row in rows}

    async def source_artist_names(self, limit: int) -> list[str]:
        """record_facts에 저장된 가수 칸 값들 (중복 제거). "A, B"처럼 여러 명이 한 칸일 수 있다."""
        rows = await self._session.execute(
            select(RecordFact.artist_name)
            .where(RecordFact.artist_name.is_not(None))
            .distinct()
            .limit(limit)
        )
        return [row.artist_name for row in rows]

    async def known_keys(self, version: str) -> set[str]:
        """이 버전으로 이미 변환한(모른다고 답한 것 포함) 가수의 name_key."""
        rows = await self._session.execute(
            select(ArtistAlias.name_key).where(ArtistAlias.alias_version == version)
        )
        return {row.name_key for row in rows}

    async def save(
        self, *, name_key: str, artist_name: str, aliases: Sequence[str], version: str
    ) -> None:
        """같은 가수가 이미 있으면 덮어쓴다 (멱등). 빈 목록도 저장한다 — 모른다는 결과다."""
        stmt = insert(ArtistAlias).values(
            name_key=name_key,
            artist_name=artist_name[:200],
            aliases=list(aliases),
            alias_version=version,
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[ArtistAlias.name_key],
            set_={
                "artist_name": stmt.excluded.artist_name,
                "aliases": stmt.excluded.aliases,
                "alias_version": stmt.excluded.alias_version,
                "updated_at": func.now(),
            },
        )
        await self._session.execute(stmt)
        await self._session.commit()
