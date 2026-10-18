import random

import pytest

from app.components.saved_song_pool import SavedSongPool
from app.db.repositories.track_mood_repository import SavedSong


class StubSearch:
    def __init__(self, songs: list[SavedSong] | None = None, error: Exception | None = None):
        self._songs = songs or []
        self._error = error
        self.calls: list[tuple[list[str], list[str], int]] = []

    async def search_by_tags(self, genres, moods, limit):
        self.calls.append((genres, moods, limit))
        if self._error:
            raise self._error
        return self._songs


def _saved(track_id, genre="댄스", moods=("신나는",)):
    return SavedSong(external_track_id=track_id, genre=genre, moods=tuple(moods))


def _pool(*songs, seed=0):
    return SavedSongPool(StubSearch(list(songs)), rng=random.Random(seed))


async def test_picks_songs_whose_genre_or_moods_match():
    pool = _pool(
        _saved("t_genre", genre="발라드", moods=("신나는",)),
        _saved("t_moods", genre="댄스", moods=("잔잔한", "그리운")),
        _saved("t_none", genre="댄스", moods=("신나는",)),
    )

    picked = await pool.pick(("발라드",), ("잔잔한", "그리운"), count=5)

    assert {s.external_track_id for s in picked} == {"t_genre", "t_moods"}


async def test_excludes_ids_the_caller_already_used():
    pool = _pool(_saved("t1", genre="발라드"), _saved("t2", genre="발라드"))

    picked = await pool.pick(("발라드",), ("신나는",), count=5, exclude_ids={"t1"})

    assert [s.external_track_id for s in picked] == ["t2"]


async def test_returns_at_most_count():
    pool = _pool(*[_saved(f"t{i}", genre="발라드") for i in range(5)])

    picked = await pool.pick(("발라드",), ("신나는",), count=2)

    assert len(picked) == 2


@pytest.mark.parametrize("count", [0, -1])
async def test_no_db_call_when_nothing_is_asked_for(count):
    search = StubSearch([_saved("t1")])
    pool = SavedSongPool(search)

    assert await pool.pick(("댄스",), ("신나는",), count=count) == []
    assert search.calls == []


async def test_db_failure_falls_back_to_empty_so_the_song_pool_covers_it():
    pool = SavedSongPool(StubSearch(error=RuntimeError("DB 장애")))

    assert await pool.pick(("댄스",), ("신나는",), count=3) == []


async def test_same_situation_does_not_always_give_the_same_songs():
    songs = [_saved(f"t{i}", genre="댄스") for i in range(12)]
    seen = set()
    for seed in range(8):
        pool = SavedSongPool(StubSearch(songs), rng=random.Random(seed))
        picked = await pool.pick(("댄스",), ("신나는",), count=2)
        seen.add(tuple(s.external_track_id for s in picked))

    assert len(seen) > 1


def _embedded(track_id, embedding, genre="발라드"):
    return SavedSong(
        external_track_id=track_id, genre=genre, moods=("잔잔한",), embedding=tuple(embedding)
    )


async def test_vector_puts_the_closest_saved_song_first(monkeypatch):
    import app.components.saved_song_pool as sp

    monkeypatch.setattr(sp, "TOP_K", 1)
    pool = _pool(_embedded("far", (0.0, 1.0)), _embedded("near", (1.0, 0.0)))

    picked = await pool.pick(("발라드",), ("잔잔한",), count=1, situation_vector=[1.0, 0.0])

    assert [s.external_track_id for s in picked] == ["near"]


async def test_saved_songs_without_vectors_or_with_other_dimension_still_work():
    pool = _pool(_saved("t1", genre="발라드"), _embedded("t2", (1.0, 0.0, 0.0)))

    picked = await pool.pick(("발라드",), ("잔잔한",), count=5, situation_vector=[1.0, 0.0])

    assert {s.external_track_id for s in picked} == {"t1", "t2"}
