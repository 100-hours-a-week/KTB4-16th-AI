"""사용자들이 저장한 곡을 추천 후보로 쓴다 (기능1·3) — 곡 풀의 두 번째 출처.

곡 풀(song_pool.py)은 TJ 노래방 차트로 미리 만든 고정 목록이라 늘지 않는다. 반면
track_moods는 사용자가 자물쇠에 곡을 저장할 때마다 기능4가 장르·무드·설명·벡터를
붙여서 쌓아 둔다 — 즉 "실제로 사람들이 고른 곡"이 저절로 모이는 곡 풀이다. 둘을
따로 두면 곡 풀이 두 개로 흩어지므로, 같은 점수 기준(song_pool.score_tags)으로
묶어서 한 풀처럼 쓴다.

여기서 뽑은 곡은 이미 Spotify ID가 있다 — LLM 곡처럼 검색(track:/artist:)으로 실존을
확인할 필요가 없고, 표시용 정보만 get_track으로 채우면 된다(TrackResolver.resolve_saved).
그래서 Spotify 호출이 곡당 1번으로 줄어든다.
"""

import logging
import random

from app.components.song_pool import (
    MIN_SCORE,
    TOP_K,
    cosine_rank_usable,
    cosine_similarity,
    score_tags,
)
from app.db.repositories.track_mood_repository import SavedSong, SavedSongSearch

logger = logging.getLogger("muro.saved_song_pool")

# 점수를 매길 후보를 몇 개까지 읽을지 — 태그가 맞는 곡이 많아도 이 안에서 고른다
SEARCH_LIMIT = 300


class SavedSongPool:
    def __init__(self, search: SavedSongSearch, rng: random.Random | None = None):
        self._search = search
        self._rng = rng or random.Random()

    async def ranked(
        self,
        situation_genres: tuple[str, ...],
        situation_moods: tuple[str, ...],
        exclude_ids: set[str] | None = None,
        situation_vector: list[float] | None = None,
    ) -> list[tuple[SavedSong, float, float | None]]:
        """태그 기준(MIN_SCORE)을 넘는 저장된 곡을 (곡, 태그 점수, 벡터 유사도)로 돌려준다.

        곡 풀의 ranked()와 같은 모양이라 db_picker가 두 출처를 같은 점수로 겨루게 한다.
        DB 조회가 실패하면 빈 목록 — 곡 풀이 채운다.
        """
        try:
            candidates = await self._search.search_by_tags(
                list(situation_genres), list(situation_moods), SEARCH_LIMIT
            )
        except Exception:
            logger.exception("저장된 곡 조회 실패 — 곡 풀만 사용")
            return []
        excluded = exclude_ids or set()
        out: list[tuple[SavedSong, float, float | None]] = []
        for song in candidates:
            if song.external_track_id in excluded:
                continue
            tag = score_tags(situation_genres, situation_moods, (song.genre,), song.moods)
            if tag < MIN_SCORE:
                continue
            sim = None
            if situation_vector is not None and cosine_rank_usable(
                situation_vector, song.embedding
            ):
                sim = cosine_similarity(situation_vector, song.embedding)
            out.append((song, tag, sim))
        return out

    async def pick(
        self,
        situation_genres: tuple[str, ...],
        situation_moods: tuple[str, ...],
        count: int,
        exclude_ids: set[str] | None = None,
        situation_vector: list[float] | None = None,
    ) -> list[SavedSong]:
        """상황에 맞는 저장된 곡 count개. 없으면 빈 목록 — 그만큼 곡 풀이 채운다.

        곡 풀과 같은 순서로 고른다: 태그로 후보를 거르고, 곡에 벡터가 있으면 상황 벡터와
        가까운 순으로 줄 세운다 (벡터가 없거나 차원이 다르면 태그 점수 순).

        부가 출처라서, DB 조회가 실패해도 추천 전체를 실패시키지 않고 빈 목록을 준다.
        """
        if count <= 0:
            return []
        try:
            candidates = await self._search.search_by_tags(
                list(situation_genres), list(situation_moods), SEARCH_LIMIT
            )
        except Exception:
            logger.exception("저장된 곡 조회 실패 — 곡 풀만 사용")
            return []

        excluded = exclude_ids or set()
        scored = [
            (score, song)
            for song in candidates
            if song.external_track_id not in excluded
            and (score := score_tags(situation_genres, situation_moods, (song.genre,), song.moods))
            >= MIN_SCORE
        ]
        if situation_vector is not None and any(
            cosine_rank_usable(situation_vector, song.embedding) for _, song in scored
        ):
            scored = [
                (cosine_similarity(situation_vector, song.embedding), song) for _, song in scored
            ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = [song for _, song in scored[:TOP_K]]
        # 같은 상황에 매번 같은 곡이 나오지 않게 상위 후보 중에서 무작위로
        self._rng.shuffle(top)
        return top[:count]
