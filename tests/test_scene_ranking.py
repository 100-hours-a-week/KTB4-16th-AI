import json

import numpy as np

from app.components.song_curator import SongCandidate
from app.components.song_pool import PoolSong, SongPool, load_song_pool
from tests.scene_fakes import song


def _pool(songs):
    return SongPool(songs)


def test_rank_scenes_matches_the_experiment_formula():
    """실험(query_style_test.py)과 같은 식: 장면마다 z점수(곡 전체 기준) → 0.5:0.5 → 장면 평균."""
    rng = np.random.default_rng(0)
    lyric = rng.normal(size=(12, 8)).astype(np.float32)
    desc = rng.normal(size=(12, 8)).astype(np.float32)
    ql = rng.normal(size=(3, 8)).astype(np.float32)
    qd = rng.normal(size=(3, 8)).astype(np.float32)
    songs = [song(f"가수{i}", f"곡{i}", vector=lyric[i], desc=desc[i]) for i in range(12)]

    ranked = _pool(songs).rank_scenes(ql.tolist(), qd.tolist())

    def z(s):
        return (s - s.mean(0)) / (s.std(0) + 1e-9)

    expected = (0.5 * z(lyric @ ql.T) + 0.5 * z(desc @ qd.T)).mean(1)
    got = {r.song.title: r.score for r in ranked}
    for i in range(12):
        assert abs(got[f"곡{i}"] - expected[i]) < 1e-4
    assert [r.song.title for r in ranked] == [f"곡{i}" for i in np.argsort(-expected)]


def test_rank_scenes_reports_the_scene_each_song_matched_best():
    songs = [
        song("가", "첫째", vector=(1, 0, 0), desc=(1, 0, 0)),
        song("나", "둘째", vector=(0, 1, 0), desc=(0, 1, 0)),
        song("다", "셋째", vector=(0, 0, 1), desc=(0, 0, 1)),
    ]
    queries = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    ranked = _pool(songs).rank_scenes(queries, queries)

    assert {r.song.title: r.scene_index for r in ranked} == {"첫째": 0, "둘째": 1, "셋째": 2}


def test_rank_scenes_uses_the_other_vector_when_a_song_has_only_one():
    only_lyric = PoolSong(
        "가", "가사만", ("발라드",), (), lyric_embedding=np.array([1, 0, 0], dtype=np.float32)
    )
    only_desc = PoolSong(
        "나", "설명만", ("발라드",), (), description_embedding=np.array([1, 0, 0], dtype=np.float32)
    )
    both = song("다", "둘다", vector=(0, 1, 0))
    q = [[1, 0, 0]]

    ranked = _pool([only_lyric, only_desc, both]).rank_scenes(q, q)

    titles = [r.song.title for r in ranked]
    assert set(titles) == {"가사만", "설명만", "둘다"}
    assert titles[-1] == "둘다"


def test_rank_scenes_skips_songs_without_any_vector_and_excluded_songs():
    nothing = PoolSong("가", "벡터없음", ("발라드",), ())
    keep = song("나", "남는곡")
    skip = song("다", "빼는곡")
    q = [[1, 0, 0]]

    ranked = _pool([nothing, keep, skip]).rank_scenes(
        q, q, exclude=[SongCandidate("다", "빼는곡", "", ())]
    )

    assert [r.song.title for r in ranked] == ["남는곡"]


def test_rank_scenes_returns_empty_when_dimensions_differ_or_queries_mismatch():
    pool = _pool([song("가", "곡")])

    assert pool.rank_scenes([[1, 0]], [[1, 0]]) == []
    assert pool.rank_scenes([[1, 0, 0]], [[1, 0, 0], [0, 1, 0]]) == []
    assert pool.rank_scenes([], []) == []
    assert _pool([]).rank_scenes([[1, 0, 0]], [[1, 0, 0]]) == []


def test_load_song_pool_reads_both_vectors_and_song_id(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(
            {
                "model": "kure",
                "dim": 3,
                "songs": [
                    {
                        "song_id": "11",
                        "artist": "가수",
                        "title": "곡",
                        "genre": "발라드",
                        "genres": ["발라드"],
                        "lyric_embedding": [1, 0, 0],
                        "description_embedding": [0, 1, 0],
                    },
                    {
                        "song_id": "12",
                        "artist": "가수2",
                        "title": "곡2",
                        "genres": ["댄스"],
                        "lyric_embedding": [1, 0, 0],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    pool = load_song_pool(path)

    first, second = pool._songs
    assert first.song_id == "11"
    assert first.lyric_embedding.dtype == np.float32
    assert first.description_embedding.tolist() == [0.0, 1.0, 0.0]
    assert first.embedding is first.lyric_embedding  # 예전 방식은 가사 벡터를 쓴다
    assert second.description_embedding is None


def test_hub_weight_subtracts_the_song_hub_value_from_the_score():
    songs = [
        song("가", "허브곡", vector=(1, 0, 0), hub=2.0),
        song("나", "보통곡", vector=(0.9, 0.1, 0), hub=0.0),
        song("다", "먼곡", vector=(0, 1, 0), hub=0.0),
    ]
    q = [[1, 0, 0]]
    pool = _pool(songs)

    plain = {r.song.title: r.score for r in pool.rank_scenes(q, q)}
    fixed = {r.song.title: r.score for r in pool.rank_scenes(q, q, hub_weight=0.5)}

    assert abs((plain["허브곡"] - fixed["허브곡"]) - 1.0) < 1e-5  # 0.5 × 2.0
    assert abs(plain["보통곡"] - fixed["보통곡"]) < 1e-6
    assert [r.song.title for r in pool.rank_scenes(q, q)][0] == "허브곡"
    assert [r.song.title for r in pool.rank_scenes(q, q, hub_weight=1.0)][0] == "보통곡"


def test_load_song_pool_reads_the_hub_value(tmp_path):
    path = tmp_path / "pool.json"
    path.write_text(
        json.dumps(
            {
                "songs": [
                    {
                        "artist": "가",
                        "title": "곡",
                        "genres": ["발라드"],
                        "lyric_embedding": [1, 0, 0],
                        "hub": 1.5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_song_pool(path)._songs[0].hub == 1.5
