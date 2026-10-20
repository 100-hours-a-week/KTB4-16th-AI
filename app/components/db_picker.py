"""DB 곡(곡 풀 + 사용자가 저장한 곡)을 같은 점수로 겨뤄서 고른다 — 기능1·3 공통.

곡 풀과 저장된 곡은 소개 문장을 같은 모델(KURE)로 벡터로 만들어 두었으므로 한 상황 벡터와의 유사도를
똑같이 잴 수 있다. 그래서 "저장된 곡 1곡 + 곡 풀 3곡"처럼 자리를 나누지 않고 점수로 겨룬다.

  점수 = 태그 가중치 × 태그 점수 + 벡터 가중치 × 벡터 유사도   (둘 다 0~1)

태그 점수는 에너지가 맞지 않는 곡이 끼는 것을 막고(9/26에 임베딩만으로 순서를 매겼다가 신나는
사진에 조용한 발라드가 1등으로 나온 적이 있다), 벡터는 그 안에서 분위기가 더 가까운 곡을
위로 올린다.
벡터가 없는 곡은 태그 점수만 쓴다. Claude가 추천한 곡은 벡터가 없어 이 경쟁에 넣지 않는다.

고르는 규칙: 점수 상위 TOP_K 안에서 무작위로(같은 상황에 매번 같은 곡이 나오지 않게), 곡 풀의 같은
가수는 1곡만, 저장된 곡은 min_saved~max_saved곡 (저장된 곡이 적은 초반에 같은 곡만 돌지 않게 상한,
실제 사람들이 고른 곡을 섞기 위해 하한).
"""

import random
from dataclasses import dataclass, replace

from app.components.song_curator import SongCandidate
from app.components.song_pool import PoolSong, norm_artist
from app.db.repositories.track_mood_repository import SavedSong

# 점수 상위 몇 곡 중에서 무작위로 고를지. 점수가 의미 있으니 고를 곡 수의 2배 정도로 좁게 둔다
# (넓게 두면 점수가 낮은 곡이 자주 뽑힌다). 같은 상황에 매번 같은 곡이 나오지 않을 만큼만 섞는다
TOP_K = 8


@dataclass(frozen=True)
class DbPickConfig:
    tag_weight: float = 0.5
    vector_weight: float = 0.5
    min_saved: int = 1
    max_saved: int = 2
    top_k: int = TOP_K


def hybrid_score(tag: float, sim: float | None, tag_weight: float, vector_weight: float) -> float:
    """태그 점수와 벡터 유사도를 가중 평균한다. 벡터가 없으면 태그 점수만."""
    if sim is None:
        return tag
    total = tag_weight + vector_weight
    if total <= 0:
        return tag
    return (tag_weight * tag + vector_weight * max(0.0, min(1.0, sim))) / total


@dataclass(frozen=True)
class _Item:
    score: float
    saved: SavedSong | None = None
    pool: PoolSong | None = None


def pick_db_songs(
    saved: list[tuple[SavedSong, float, float | None]],
    pool: list[tuple[PoolSong, float, float | None]],
    count: int,
    config: DbPickConfig = DbPickConfig(),
    rng: random.Random | None = None,
) -> tuple[list[SavedSong], list[SongCandidate]]:
    """저장된 곡·곡 풀 후보를 같은 점수로 겨뤄 count곡을 고른다. 고른 곡에는 score를 붙인다."""
    if count <= 0:
        return [], []
    rng = rng or random.Random()
    w_tag, w_vec = config.tag_weight, config.vector_weight
    items = [_Item(hybrid_score(tag, sim, w_tag, w_vec), saved=s) for s, tag, sim in saved] + [
        _Item(hybrid_score(tag, sim, w_tag, w_vec), pool=p) for p, tag, sim in pool
    ]
    items.sort(key=lambda it: (it.score, it.pool.popularity if it.pool else 0), reverse=True)
    max_saved = max(config.max_saved, 0)

    top = items[: config.top_k]
    rng.shuffle(top)

    picked: list[_Item] = []
    artists: set[str] = set()
    saved_count = 0

    def take(it: _Item) -> bool:
        """저장된 곡 상한·같은 가수 1곡 규칙을 지키면 고르고 True."""
        nonlocal saved_count
        if it.saved is not None:
            if saved_count >= max_saved:
                return False
            saved_count += 1
        else:
            artist = norm_artist(it.pool.artist)
            if artist in artists:
                return False
            artists.add(artist)
        picked.append(it)
        return True

    for it in top:
        take(it)
        if len(picked) == count:
            break
    # 같은 가수·저장 곡 상한 때문에 모자라면 상위 밖의 후보를 점수 순으로 이어서 채운다
    for it in items[config.top_k :]:
        if len(picked) == count:
            break
        take(it)

    # 저장된 곡이 하한보다 적으면 (점수가 가장 높은 저장된 곡으로) 가장 낮은 곡 풀 곡을 바꾼다
    spare_saved = [it for it in items if it.saved is not None and it not in picked]
    while saved_count < min(config.min_saved, max_saved) and spare_saved:
        pool_picks = [it for it in picked if it.pool is not None]
        if len(picked) < count:
            picked.append(spare_saved.pop(0))
        elif pool_picks:
            worst = min(pool_picks, key=lambda it: it.score)
            picked[picked.index(worst)] = spare_saved.pop(0)
        else:
            break
        saved_count += 1

    saved_out = [replace(it.saved, score=it.score) for it in picked if it.saved is not None]
    pool_out = [replace(it.pool.to_candidate(), score=it.score) for it in picked if it.pool]
    return saved_out, pool_out
