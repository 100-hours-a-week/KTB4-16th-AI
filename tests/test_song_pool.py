import json
import random
from pathlib import Path

import pytest

from app.components.song_curator import SongCandidate
from app.components.song_pool import PoolSong, SongPool, load_song_pool
from app.config import get_settings


def _pool(*songs: PoolSong, seed: int = 0) -> SongPool:
    return SongPool(list(songs), rng=random.Random(seed))


def _ps(artist, title, genres=("발라드",), moods=("잔잔한",), popularity=0) -> PoolSong:
    return PoolSong(artist, title, tuple(genres), tuple(moods), popularity)


def test_picks_only_songs_matching_genre_or_enough_moods():
    pool = _pool(
        _ps("A", "장르 맞음", genres=("발라드",), moods=("신나는",)),
        _ps("B", "무드 다 맞음", genres=("댄스",), moods=("잔잔한", "그리운")),
        _ps("C", "무드 하나만", genres=("댄스",), moods=("잔잔한",)),
        _ps("D", "아무것도 안 맞음", genres=("댄스",), moods=("신나는",)),
    )

    picked = pool.pick(("발라드",), ("잔잔한", "그리운"), count=5)

    titles = {s.title for s in picked}
    # 무드 하나만 겹치면 0.25점 — 0.5 미만이라 후보가 아니다
    assert titles == {"장르 맞음", "무드 다 맞음"}


def test_secondary_genre_counts_as_match():
    # 한로로처럼 주 장르는 록이고 인디음악이 두 번째로 붙은 곡도 인디 상황에 잡혀야 한다
    pool = _pool(_ps("한로로", "입춘", genres=("록/메탈", "인디음악"), moods=("차분한",)))

    picked = pool.pick(("인디음악",), ("잔잔한",), count=3)

    assert [s.title for s in picked] == ["입춘"]
    assert picked[0].genre == "록/메탈"


def test_excludes_songs_the_llm_already_picked_ignoring_spacing_and_alias():
    pool = _pool(
        _ps("RESCENE(리센느)", "LOVE ATTACK"),
        _ps("다른 가수", "다른 곡"),
    )
    llm = [SongCandidate("rescene", "LOVE  ATTACK", "댄스", ())]

    picked = pool.pick(("발라드",), ("잔잔한",), count=3, exclude=llm)

    assert [s.title for s in picked] == ["다른 곡"]


def test_one_song_per_artist():
    pool = _pool(_ps("A", "1"), _ps("A", "2"), _ps("A", "3"), _ps("B", "1"))

    picked = pool.pick(("발라드",), ("잔잔한",), count=3)

    assert sorted(s.artist for s in picked) == ["A", "B"]


def test_random_among_top_so_same_situation_varies():
    pool = SongPool([_ps(f"가수{i}", f"곡{i}") for i in range(15)], rng=random.Random(7))

    seen = {tuple(s.title for s in pool.pick(("발라드",), ("잔잔한",), count=3)) for _ in range(20)}

    assert len(seen) > 1


def test_zero_count_or_no_match_returns_empty():
    pool = _pool(_ps("A", "1", genres=("댄스",), moods=("신나는",)))

    assert pool.pick(("발라드",), ("잔잔한",), count=0) == []
    assert pool.pick(("발라드",), ("잔잔한",), count=3) == []


def test_load_skips_songs_without_moods_and_handles_broken_file(tmp_path):
    good = tmp_path / "pool.json"
    good.write_text(
        json.dumps(
            {
                "songs": [
                    {
                        "artist": "A",
                        "title": "1",
                        "genres": ["발라드"],
                        "moods": ["잔잔한"],
                        "popularity": 50,
                    },
                    {"artist": "B", "title": "2", "genres": ["발라드"], "moods": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")

    pool = load_song_pool(good)

    assert pool is not None and len(pool) == 1
    assert load_song_pool(broken) is None
    assert load_song_pool(tmp_path / "없음.json") is None


def test_downloaded_pool_uses_only_known_vocab():
    """빌드 때 HF에서 받은 곡 풀이 있으면 어휘가 맞는지 본다 (공개 레포에는 파일이 없다)."""
    from app.components.music_tags import GENRE_VOCAB, MOOD_VOCAB

    path = Path(get_settings().song_pool_path)
    if not path.exists():
        pytest.skip("곡 풀 파일 없음 (비공개 HF 저장소에서 받음)")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["songs"]) > 500
    for s in data["songs"]:
        assert set(s["genres"]) <= set(GENRE_VOCAB), s
        assert set(s["moods"]) <= set(MOOD_VOCAB), s
        assert 0 <= s["popularity"] <= 100
