"""LLM 곡·곡 풀 곡·사용자 저장 곡을 같이 Spotify에서 확인한다 — 기능1·기능3 공통.

여러 목록을 동시에 확인하고, 한쪽 확인만 실패하면 다른 쪽으로 계속한다. 확인을 시도한
쪽이 모두 실패하면 Spotify 장애로 보고 UpstreamError를 그대로 던진다(서비스가 502로).

저장 곡(track_moods)은 곡 풀과 같은 한 풀로 취급하므로 pool 쪽 결과에 합쳐서 돌려준다.
저장 곡 조회는 부가 출처라 실패해도 전체를 실패시키지 않는다(resolve_saved가 흡수).
"""

import asyncio
import logging

from app.components.reranker import TrackCandidate
from app.components.song_curator import SongCandidate
from app.components.track_resolver import TrackResolver
from app.db.repositories.track_mood_repository import SavedSong
from app.exceptions import UpstreamError

logger = logging.getLogger("muro.song_mix")


async def resolve_both(
    resolver: TrackResolver, llm_songs: list[SongCandidate], pool_songs: list[SongCandidate]
) -> tuple[list[TrackCandidate], list[TrackCandidate]]:
    llm_result, pool_result = await asyncio.gather(
        resolver.resolve(llm_songs) if llm_songs else _empty(),
        resolver.resolve(pool_songs) if pool_songs else _empty(),
        return_exceptions=True,
    )
    tried = [r for r, songs in ((llm_result, llm_songs), (pool_result, pool_songs)) if songs]
    errors = [r for r in tried if isinstance(r, BaseException)]
    for e in errors:
        if not isinstance(e, UpstreamError):
            raise e
    if errors and len(errors) == len(tried):
        raise errors[0]
    if errors:
        logger.warning("Spotify 확인 일부 실패, 나머지로 응답: %s", errors[0])
    llm_found = [] if isinstance(llm_result, BaseException) else llm_result
    pool_found = [] if isinstance(pool_result, BaseException) else pool_result
    return llm_found, pool_found


async def _empty() -> list[TrackCandidate]:
    return []


async def resolve_sources(
    resolver: TrackResolver,
    llm_songs: list[SongCandidate],
    pool_songs: list[SongCandidate],
    saved_songs: list[SavedSong],
) -> tuple[list[TrackCandidate], list[TrackCandidate]]:
    """(LLM 곡, 곡 풀 곡 + 저장 곡)으로 돌려준다. 저장 곡은 곡 풀과 한 풀로 합친다."""
    saved_found, resolved = await asyncio.gather(
        resolver.resolve_saved(saved_songs),
        resolve_both(resolver, llm_songs, pool_songs),
        return_exceptions=True,
    )
    if isinstance(resolved, BaseException):
        raise resolved
    if isinstance(saved_found, BaseException):
        logger.warning("저장 곡 확인 실패, 곡 풀만 사용: %s", saved_found)
        saved_found = []
    llm_found, pool_found = resolved
    return llm_found, [*saved_found, *pool_found]
