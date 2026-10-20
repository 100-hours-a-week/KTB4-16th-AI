import random

from app.components.db_picker import DbPickConfig, hybrid_score, pick_db_songs
from app.components.song_pool import PoolSong
from app.db.repositories.track_mood_repository import SavedSong


def _pool(artist, title, tag=1.0, sim=None, pop=0):
    return (PoolSong(artist, title, ("발라드",), ("잔잔한",), pop), tag, sim)


def _saved(track_id, tag=1.0, sim=None):
    return (SavedSong(track_id, "발라드", ("잔잔한",)), tag, sim)


def _pick(saved, pool, count=4, **cfg):
    cfg.setdefault(
        "top_k", count
    )  # 무작위로 섞이는 범위를 고를 곡 수와 같게 — 점수 순서대로 뽑힌다
    config = DbPickConfig(**cfg)
    return pick_db_songs(saved, pool, count, config, rng=random.Random(0))


def test_hybrid_score_mixes_tag_and_vector_and_falls_back_to_tag_only():
    assert hybrid_score(1.0, 0.5, 0.5, 0.5) == 0.75
    assert hybrid_score(0.5, None, 0.5, 0.5) == 0.5  # 벡터가 없으면 태그 점수만
    assert hybrid_score(0.0, -0.3, 0.5, 0.5) == 0.0  # 음수 유사도는 0으로


def test_saved_and_pool_songs_compete_on_the_same_score():
    saved = [_saved(f"s{i}", sim=0.95) for i in range(3)]  # 분위기가 아주 가까운 저장된 곡
    pool = [_pool(f"가수{i}", f"곡{i}", sim=0.2) for i in range(4)]

    saved_out, pool_out = _pick(saved, pool, count=4, max_saved=3)

    assert len(saved_out) == 3 and len(pool_out) == 1  # 점수가 높은 저장된 곡이 더 많이 들어간다


def test_max_saved_caps_how_many_saved_songs_get_in():
    saved = [_saved(f"s{i}", sim=0.95) for i in range(5)]
    pool = [_pool(f"가수{i}", f"곡{i}", sim=0.2) for i in range(5)]

    saved_out, _ = _pick(saved, pool, count=4, min_saved=1, max_saved=2)

    assert len(saved_out) <= 2  # 점수가 가장 높아도 상한을 넘지 않는다


def test_min_saved_swaps_in_a_saved_song_even_when_pool_songs_score_higher():
    saved = [_saved("s1", tag=0.5, sim=0.1)]  # 점수가 낮은 저장된 곡
    pool = [_pool(f"가수{i}", f"곡{i}", sim=0.9) for i in range(6)]

    saved_out, pool_out = _pick(saved, pool, count=4, min_saved=1, max_saved=2)

    assert [s.external_track_id for s in saved_out] == ["s1"] and len(pool_out) == 3


def test_same_artist_is_picked_only_once_from_the_pool():
    pool = [_pool("같은가수", f"곡{i}", sim=0.9) for i in range(4)] + [
        _pool("다른가수", "곡", sim=0.5)
    ]

    _, pool_out = _pick([], pool, count=4)

    assert sorted(s.artist for s in pool_out) == ["같은가수", "다른가수"]


def test_picked_songs_carry_their_score_for_the_final_order():
    saved_out, pool_out = _pick([_saved("s1", sim=0.8)], [_pool("가수", "곡", sim=0.6)], count=2)

    assert saved_out[0].score == hybrid_score(1.0, 0.8, 0.5, 0.5)
    assert pool_out[0].score == hybrid_score(1.0, 0.6, 0.5, 0.5)


def test_zero_count_picks_nothing():
    assert _pick([_saved("s1")], [_pool("가수", "곡")], count=0) == ([], [])


def test_fills_up_from_outside_the_top_when_artist_rule_leaves_it_short():
    pool = [_pool("같은가수", f"곡{i}", sim=0.9) for i in range(6)] + [
        _pool(f"가수{i}", "곡", sim=0.1) for i in range(5)
    ]

    _, pool_out = _pick([], pool, count=4)

    assert len(pool_out) == 4 and len({s.artist for s in pool_out}) == 4
