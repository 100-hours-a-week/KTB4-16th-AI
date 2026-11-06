import random

from app.components.scene_picker import MIN_TASTE_CANDIDATES, pick_scene_songs
from app.components.song_pool import ScoredSong
from tests.scene_fakes import song


def _ranked(
    genre: str, count: int, start_score: float, prefix: str | None = None
) -> list[ScoredSong]:
    prefix = prefix or genre
    return [
        ScoredSong(song(f"{prefix}가수{i}", f"{prefix}곡{i}", genre), start_score - i * 0.01, i % 3)
        for i in range(count)
    ]


def _sorted(*groups) -> list[ScoredSong]:
    return sorted([r for g in groups for r in g], key=lambda r: r.score, reverse=True)


def test_without_taste_picks_from_the_top_k_only():
    ranked = _ranked("발라드", 100, 3.0)

    picks = pick_scene_songs(ranked, 5, None, top_k=20, rng=random.Random(1))

    assert len(picks) == 5
    top_titles = {r.song.title for r in ranked[:20]}
    assert {p.song.title for p in picks} <= top_titles
    assert not any(p.outside_taste for p in picks)


def test_picks_vary_with_the_random_source():
    ranked = _ranked("발라드", 100, 3.0)

    seen = {
        tuple(p.song.title for p in pick_scene_songs(ranked, 5, None, rng=random.Random(s)))
        for s in range(8)
    }

    assert len(seen) > 1


def test_taste_filter_keeps_only_taste_genres_plus_one_outside():
    ranked = _sorted(
        _ranked("발라드", 60, 2.0), _ranked("댄스", 60, 3.0), _ranked("록/메탈", 30, 4.0)
    )

    picks = pick_scene_songs(ranked, 5, ["발라드"], rng=random.Random(2))

    genres = [p.song.genres[0] for p in picks]
    assert genres.count("발라드") == 4
    outside = [p for p in picks if p.outside_taste]
    assert len(outside) == 1
    assert outside[0].song.genres[0] in {"댄스", "록/메탈"}


def test_outside_pick_comes_from_the_best_few_outside_the_taste():
    ranked = _sorted(_ranked("발라드", 60, 2.0), _ranked("댄스", 60, 3.0))
    best_outside = {r.song.title for r in ranked if r.song.genres[0] == "댄스"}
    best_outside = {r.song.title for r in [r for r in ranked if r.song.genres[0] == "댄스"][:5]}

    for seed in range(10):
        picks = pick_scene_songs(ranked, 5, ["발라드"], rng=random.Random(seed))
        outside = next(p for p in picks if p.outside_taste)
        assert outside.song.title in best_outside


def test_outside_position_places_the_outside_song():
    ranked = _sorted(_ranked("발라드", 60, 2.0), _ranked("댄스", 60, 3.0))

    picks = pick_scene_songs(ranked, 6, ["발라드"], outside_position=2, rng=random.Random(3))

    assert picks[2].outside_taste is True
    assert sum(p.outside_taste for p in picks) == 1


def test_default_outside_position_is_last():
    ranked = _sorted(_ranked("발라드", 60, 2.0), _ranked("댄스", 60, 3.0))

    picks = pick_scene_songs(ranked, 5, ["발라드"], rng=random.Random(4))

    assert picks[-1].outside_taste is True


def test_too_few_candidates_in_the_taste_widens_to_everything():
    few = MIN_TASTE_CANDIDATES - 1
    ranked = _sorted(_ranked("발라드", few, 2.0), _ranked("댄스", 80, 3.0))

    picks = pick_scene_songs(ranked, 5, ["발라드"], rng=random.Random(5))

    assert not any(p.outside_taste for p in picks)
    assert {p.song.genres[0] for p in picks} == {"댄스"}  # 전체에서 점수 상위 → 댄스


def test_unknown_taste_genre_widens_to_everything():
    ranked = _ranked("발라드", 80, 2.0)

    picks = pick_scene_songs(ranked, 5, ["트로트"], rng=random.Random(6))

    assert len(picks) == 5
    assert not any(p.outside_taste for p in picks)


def test_same_artist_is_picked_once_and_the_list_is_filled_beyond_top_k():
    ranked = [
        ScoredSong(song("같은가수", f"곡{i}", "발라드"), 3.0 - i * 0.01, 0) for i in range(30)
    ] + _ranked("발라드", 30, 1.0)

    picks = pick_scene_songs(ranked, 5, None, top_k=10, rng=random.Random(7))

    assert len(picks) == 5
    assert [p.song.artist for p in picks].count("같은가수") == 1


def test_returns_what_exists_when_the_pool_is_small():
    ranked = _ranked("발라드", 3, 2.0)

    assert len(pick_scene_songs(ranked, 5, None, rng=random.Random(8))) == 3


def test_empty_input_and_zero_count():
    assert pick_scene_songs([], 5, None) == []
    assert pick_scene_songs(_ranked("발라드", 5, 1.0), 0, None) == []
