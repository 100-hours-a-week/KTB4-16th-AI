from app.components.taste_genres import LOCK_GENRE_LIMIT, no_onboarding, taste_genres


class Records:
    def __init__(self, genres=None, error=None):
        self._genres = genres or {}
        self._error = error

    async def genres_by_record(self, user_id):
        if self._error:
            raise self._error
        return self._genres


def _onboarding(genres=None, error=None):
    async def read(user_id):
        if error:
            raise error
        return genres or []

    return read


async def test_lock_genres_by_count_then_onboarding():
    records = Records(
        {1: "발라드", 2: "발라드", 3: "발라드", 4: "인디음악", 5: "인디음악", 6: "POP"}
    )

    result = await taste_genres(7, _onboarding(["댄스"]), records)

    assert result == ["발라드", "인디음악", "POP", "댄스"]


async def test_only_the_top_lock_genres_are_used():
    records = Records(
        {i: g for i, g in enumerate(["발라드"] * 4 + ["댄스"] * 3 + ["POP"] * 2 + ["OST"])}
    )

    result = await taste_genres(7, no_onboarding, records)

    assert result == ["발라드", "댄스", "POP"]
    assert len(result) == LOCK_GENRE_LIMIT


async def test_onboarding_genres_already_in_locks_are_not_repeated():
    records = Records({1: "발라드"})

    assert await taste_genres(7, _onboarding(["발라드", "댄스"]), records) == ["발라드", "댄스"]


async def test_unknown_genres_are_ignored():
    records = Records({1: "시티팝", 2: "발라드"})

    assert await taste_genres(7, _onboarding(["로파이"]), records) == ["발라드"]


async def test_no_signals_means_empty():
    assert await taste_genres(7, no_onboarding, Records()) == []


async def test_a_failing_source_does_not_block_the_other():
    broken_locks = Records(error=RuntimeError("pg down"))
    assert await taste_genres(7, _onboarding(["댄스"]), broken_locks) == ["댄스"]

    broken_onboarding = _onboarding(error=RuntimeError("mysql down"))
    assert await taste_genres(7, broken_onboarding, Records({1: "발라드"})) == ["발라드"]
