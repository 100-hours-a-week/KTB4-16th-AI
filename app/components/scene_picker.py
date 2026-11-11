"""장면 문장으로 매긴 곡 점수(SongPool.rank_scenes)에서 최종 곡을 고른다 — 기능1·3 공통.

  1. 취향 장르가 있으면 그 장르 곡만 후보로 남긴다 (SQL WHERE처럼 점수는 건드리지 않는다).
     남은 후보가 MIN_TASTE_CANDIDATES곡보다 적으면 취향으로 거르지 않고 전체를 후보로 쓴다.
  2. 후보 중 점수 상위 TOP_K곡을 섞어서(같은 상황에 매번 같은 곡이 나오지 않게) 앞에서부터
     고른다. 같은 가수는 1곡만.
  3. 취향으로 걸렀다면 한 자리는 취향 밖 장르 곡으로 둔다 — 취향 밖 곡 중 점수 상위
     OUTSIDE_TOP곡에서 무작위로 하나. 같은 취향 안에서만 계속 돌지 않게 하려는 자리다.

숫자(TOP_K·OUTSIDE_TOP·MIN_TASTE_CANDIDATES)는 100개 상황 실측으로 정하기 전의 초기값이다.
"""

import random
from dataclasses import dataclass

from app.components.song_pool import PoolSong, ScoredSong, norm_artist

TOP_K = 20
OUTSIDE_TOP = 5
MIN_TASTE_CANDIDATES = 50


@dataclass(frozen=True)
class Pick:
    song: PoolSong
    score: float
    scene_index: int
    outside_taste: bool = False


def pick_scene_songs(
    ranked: list[ScoredSong],
    count: int,
    taste: list[str] | None = None,
    *,
    outside_position: int | None = None,
    top_k: int = TOP_K,
    outside_top: int = OUTSIDE_TOP,
    min_candidates: int = MIN_TASTE_CANDIDATES,
    rng: random.Random | None = None,
) -> list[Pick]:
    """count곡을 순서대로 돌려준다. 취향 밖 곡은 outside_position 자리에 넣는다(기본 맨 뒤)."""
    if count <= 0 or not ranked:
        return []
    rng = rng or random.Random()
    taste_set = set(taste or [])
    inside = [r for r in ranked if taste_set & set(r.song.genres)] if taste_set else []
    filtered = bool(taste_set) and len(inside) >= min_candidates
    candidates = inside if filtered else ranked
    use_outside = filtered and count >= 2

    main_count = count - 1 if use_outside else count
    artists: set[str] = set()
    picks: list[Pick] = []

    def take(r: ScoredSong, *, outside: bool = False) -> bool:
        artist = norm_artist(r.song.artist)
        if artist in artists:
            return False
        artists.add(artist)
        picks.append(Pick(r.song, r.score, r.scene_index, outside))
        return True

    top = list(candidates[:top_k])
    rng.shuffle(top)
    for r in top:
        if len(picks) == main_count:
            break
        take(r)
    # 같은 가수 때문에 모자라면 상위 밖 후보를 점수 순으로 이어서 채운다
    for r in candidates[top_k:]:
        if len(picks) == main_count:
            break
        take(r)

    if use_outside:
        pool = [r for r in ranked if not taste_set & set(r.song.genres)][:outside_top]
        rng.shuffle(pool)
        for r in pool:
            if take(r, outside=True):
                break
        else:
            # 취향 밖 곡이 없거나 가수가 겹치면 취향 안에서 한 곡 더
            for r in candidates:
                if len(picks) == count:
                    break
                take(r)
        outside = [p for p in picks if p.outside_taste]
        if outside:
            picks.remove(outside[0])
            position = len(picks) if outside_position is None else min(outside_position, len(picks))
            picks.insert(position, outside[0])
    return picks[:count]
