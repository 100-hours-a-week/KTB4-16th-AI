import random

from app.components.song_battle_generator import OPTION_MAX_LEN, TEMPLATES, generate_battle
from app.components.song_pool import PoolSong, SongPool


def _song(artist, title, genre="댄스", popularity=50):
    return PoolSong(artist, title, (genre,), ("신나는",), popularity)


def test_picks_two_songs_of_the_same_genre_within_a_similar_popularity_band():
    pool = SongPool([_song("A", "1", popularity=50), _song("B", "2", popularity=55)])

    battle = generate_battle("댄스", pool, used_pairs=set(), rng=random.Random(0))

    assert battle is not None
    assert battle.question in TEMPLATES
    assert {battle.option_a, battle.option_b} == {"A - 1", "B - 2"}


def test_does_not_pair_songs_whose_popularity_gap_is_too_wide():
    pool = SongPool([_song("A", "1", popularity=10), _song("B", "2", popularity=90)])

    assert generate_battle("댄스", pool, used_pairs=set(), rng=random.Random(0)) is None


def test_ignores_songs_from_a_different_genre():
    pool = SongPool([_song("A", "1", genre="댄스"), _song("B", "2", genre="발라드")])

    assert generate_battle("댄스", pool, used_pairs=set(), rng=random.Random(0)) is None


def test_skips_pairs_already_used_recently():
    pool = SongPool(
        [
            _song("A", "1", popularity=50),
            _song("B", "2", popularity=52),
            _song("C", "3", popularity=54),
        ]
    )
    used = {frozenset({"A - 1", "B - 2"})}

    battle = generate_battle("댄스", pool, used_pairs=used, rng=random.Random(0))

    assert battle is not None
    assert frozenset({battle.option_a, battle.option_b}) != frozenset({"A - 1", "B - 2"})


def test_no_pair_left_when_every_combination_is_used():
    pool = SongPool([_song("A", "1", popularity=50), _song("B", "2", popularity=52)])
    used = {frozenset({"A - 1", "B - 2"})}

    assert generate_battle("댄스", pool, used_pairs=used, rng=random.Random(0)) is None


def test_long_song_names_are_truncated_to_fit_the_option_column():
    long_title = "아" * 60
    pool = SongPool([_song("A", long_title, popularity=50), _song("B", "짧은곡", popularity=52)])

    battle = generate_battle("댄스", pool, used_pairs=set(), rng=random.Random(0))

    assert battle is not None
    assert max(len(battle.option_a), len(battle.option_b)) <= OPTION_MAX_LEN


def test_a_and_b_are_never_the_same_song():
    pool = SongPool([_song("A", "1", popularity=50)])

    assert generate_battle("댄스", pool, used_pairs=set(), rng=random.Random(0)) is None
