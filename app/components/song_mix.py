"""LLM 곡과 곡 풀 곡을 같이 Spotify에서 확인한다 — 기능1·기능3 공통.

두 목록을 동시에 확인하고, 한쪽 확인만 실패하면 다른 쪽으로 계속한다. 확인을 시도한 쪽이
모두 실패하면 Spotify 장애로 보고 UpstreamError를 그대로 던진다(서비스가 502로 바꾼다).
"""

import asyncio
import logging

from app.components.reranker import TrackCandidate
from app.components.song_curator import SongCandidate
from app.components.track_resolver import TrackResolver
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
