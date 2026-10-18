import random
from datetime import UTC, date, datetime

import pytest

from app.components.music_genre_assigner import SavedTrack, assign_genres
from app.components.music_tags import MUSIC_GENRE_GROUPS
from app.db.repositories.balance_stats_repository import GenreStats
from app.workers import balance_daily_worker as w
from app.workers.balance_daily_worker import GroupFailedError
from tests.conftest import AUTH

T = datetime(2026, 10, 1, tzinfo=UTC)
LATER = datetime(2026, 10, 5, tzinfo=UTC)
DAY = date(2026, 10, 17)


def _s(user, track, at=T):
    return SavedTrack(user, track, at)


GENRES = {"b1": "발라드", "b2": "발라드", "i1": "인디음악", "old": "시티팝"}


def test_top_genre_per_user():
    saved = [_s(1, "b1"), _s(1, "b2"), _s(1, "i1")]
    result = assign_genres(saved, GENRES, min_group_size=1)
    assert result[1].music_genre == "발라드"
    assert result[1].source == "records"


def test_tie_goes_to_more_recent_genre():
    saved = [_s(1, "b1", T), _s(1, "i1", LATER)]
    assert assign_genres(saved, GENRES, min_group_size=1)[1].music_genre == "인디음악"


def test_small_group_and_unknown_genre_merge_into_the_biggest_genre():
    saved = [_s(1, "b1"), _s(2, "b1"), _s(3, "i1"), _s(4, "old"), _s(5, "old2")]
    result = assign_genres(saved, GENRES, min_group_size=2)
    assert result[1].music_genre == "발라드" and result[2].music_genre == "발라드"
    assert result[1].source == "records"
    # 인디 1명뿐, 13개 밖 장르 → 공통으로 분류됐다가 가장 큰 그룹(발라드)에 합쳐진다
    assert {result[u].music_genre for u in (3, 4, 5)} == {"발라드"}
    assert {result[u].source for u in (3, 4, 5)} == {"merged"}


def test_common_users_merge_even_when_the_common_group_is_big_enough():
    # 발라드 3명, 장르 모름 3명 / 기준 2명 → 공통이 기준 이상이어도 발라드로 합침
    saved = [_s(1, "b1"), _s(2, "b1"), _s(3, "b2"), _s(4, "old"), _s(5, "old"), _s(6, "old")]
    result = assign_genres(saved, GENRES, min_group_size=2)
    assert {result[u].music_genre for u in range(1, 7)} == {"발라드"}
    assert result[4].source == "merged"


def test_common_stays_when_no_genre_group_is_big_enough():
    # 오픈 초반: 모든 장르가 기준 미만 → 다 공통 (합칠 큰 그룹이 없음)
    saved = [_s(1, "b1"), _s(2, "i1")]
    result = assign_genres(saved, GENRES, min_group_size=6)
    assert {a.music_genre for a in result.values()} == {"공통"}


def test_due_date_only_inside_batch_window():
    # 23:10 KST = 14:10 UTC → 다음 날 몫
    assert w.due_date(datetime(2026, 10, 15, 14, 10, tzinfo=UTC)) == date(2026, 10, 16)
    # 22:59 KST, 23:40 KST → 배치 시간 아님
    assert w.due_date(datetime(2026, 10, 15, 13, 59, tzinfo=UTC)) is None
    assert w.due_date(datetime(2026, 10, 15, 14, 40, tzinfo=UTC)) is None


def test_job_is_once_per_target_date():
    fields = w.job_fields(date(2026, 10, 16))
    assert fields["dedupe_key"] == "2026-10-16"
    assert fields["payload"] == {"date": "2026-10-16"}


def test_manual_trigger_queues_job(client, jobs):
    res = client.post("/api/balance/daily-batch", json={"date": "2026-10-20"}, headers=AUTH)
    assert res.status_code == 202
    assert res.json() == {"status": "QUEUED", "date": "2026-10-20"}
    assert ("balance_daily", "2026-10-20") in jobs.jobs


def test_manual_trigger_defaults_to_tomorrow(client, jobs):
    res = client.post("/api/balance/daily-batch", json={}, headers=AUTH)
    assert res.status_code == 202
    assert date.fromisoformat(res.json()["date"]) > datetime.now(UTC).date()


# ── _fill: 14개 그룹 전부 + 예비 질문 (2026-10-16) ─────────────────────────


class FakeStatsSource:
    async def get(self, genre):
        return GenreStats.empty()


class FakeGenerator:
    """generate 호출마다 candidates 목록을 하나씩 꺼내 준다."""

    model = "fake-llm"

    def __init__(self, *batches):
        self._batches = list(batches)

    async def generate(self, genre, stats, count):
        return self._batches.pop(0)


class FakeRepo:
    def __init__(self, spares=()):
        self.approved: dict[tuple, str] = {}
        self.option_pairs: dict[tuple, set] = {}
        self.spares: list[dict] = list(spares)

    async def add_question(
        self, *, question_date, music_genre, question, option_a, option_b, model_version
    ):
        key = (question_date, music_genre)
        if key in self.approved:
            return False
        self.approved[key] = question
        self.option_pairs[key] = {option_a, option_b}
        return True

    async def add_spare(
        self, *, question_date, music_genre, question, option_a, option_b, model_version
    ):
        self.spares.append({"genre": music_genre, "question": question})

    async def use_spare(self, question_date, music_genre):
        for i, spare in enumerate(self.spares):
            if spare["genre"] == music_genre:
                picked = self.spares.pop(i)
                self.approved[(question_date, music_genre)] = picked["question"]
                return picked["question"]
        return None


def test_music_genre_groups_has_13_genres_plus_common():
    """사용자 수와 상관없이 매일 채우는 그룹 전체 — 13장르 + 공통."""
    assert len(MUSIC_GENRE_GROUPS) == 14
    assert "공통" in MUSIC_GENRE_GROUPS


async def test_fill_approves_first_candidate_and_banks_the_rest_as_spare():
    repo = FakeRepo()
    generator = FakeGenerator([("질문1", "A1", "B1"), ("질문2", "A2", "B2"), ("질문3", "A3", "B3")])
    used = set()

    await w._fill(
        repo,
        generator,
        FakeStatsSource(),
        DAY,
        "발라드",
        used,
        None,
        set(),
        random.Random(),
    )

    assert repo.approved[(DAY, "발라드")] == "질문1"
    assert [s["question"] for s in repo.spares] == ["질문2", "질문3"]
    assert used == {"질문1", "질문2", "질문3"}


async def test_fill_skips_candidates_already_used_recently():
    repo = FakeRepo()
    generator = FakeGenerator([("옛 질문", "A1", "B1"), ("새 질문", "A2", "B2")])
    used = {"옛 질문"}

    await w._fill(
        repo,
        generator,
        FakeStatsSource(),
        DAY,
        "댄스",
        used,
        None,
        set(),
        random.Random(),
    )

    assert repo.approved[(DAY, "댄스")] == "새 질문"


async def test_fill_uses_a_banked_spare_when_no_fresh_candidate():
    """새 후보가 다 최근 질문과 겹치면 LLM을 또 안 부르고 쌓아 둔 예비 질문을 쓴다."""
    repo = FakeRepo(spares=[{"genre": "인디음악", "question": "예비 질문"}])
    generator = FakeGenerator([("옛 질문", "A", "B")])
    used = {"옛 질문"}

    await w._fill(
        repo,
        generator,
        FakeStatsSource(),
        DAY,
        "인디음악",
        used,
        None,
        set(),
        random.Random(),
    )

    assert repo.approved[(DAY, "인디음악")] == "예비 질문"
    assert repo.spares == []  # 한 번 쓴 예비 질문은 다시 안 쓴다
    assert "예비 질문" in used


async def test_fill_raises_when_no_fresh_candidate_and_no_spare():
    repo = FakeRepo()
    generator = FakeGenerator([("옛 질문", "A", "B")])
    used = {"옛 질문"}

    with pytest.raises(GroupFailedError):
        await w._fill(
            repo,
            generator,
            FakeStatsSource(),
            DAY,
            "재즈",
            used,
            None,
            set(),
            random.Random(),
        )


async def test_fill_does_not_reuse_a_spare_from_a_different_genre():
    repo = FakeRepo(spares=[{"genre": "트로트", "question": "트로트 예비"}])
    generator = FakeGenerator([("옛 질문", "A", "B")])
    used = {"옛 질문"}

    with pytest.raises(GroupFailedError):
        await w._fill(
            repo,
            generator,
            FakeStatsSource(),
            DAY,
            "재즈",
            used,
            None,
            set(),
            random.Random(),
        )
    assert repo.spares  # 다른 장르 예비는 그대로 남아 있다


# ── 곡 대결 우선 (song_battle_generator, 2026-10-16) ───────────────────────


def _pool_song(artist, title, genre, popularity):
    from app.components.song_pool import PoolSong

    return PoolSong(artist, title, (genre,), ("신나는",), popularity)


class ExplodingGenerator:
    """LLM이 불리면 바로 실패 — 곡 대결이 성공하면 LLM을 아예 안 불러야 한다."""

    model = "should-not-be-called"

    async def generate(self, genre, stats, count):
        raise AssertionError("곡 대결이 되는데 LLM 취향 질문을 불렀다")


async def test_fill_uses_song_battle_without_calling_the_llm():
    from app.components.song_pool import SongPool

    repo = FakeRepo()
    pool = SongPool([_pool_song("A", "곡1", "댄스", 50), _pool_song("B", "곡2", "댄스", 55)])

    await w._fill(
        repo,
        ExplodingGenerator(),
        FakeStatsSource(),
        DAY,
        "댄스",
        set(),
        pool,
        set(),
        random.Random(0),
    )

    from app.components.song_battle_generator import TEMPLATES

    key = (DAY, "댄스")
    assert repo.approved[key] in TEMPLATES
    pair = repo.option_pairs[key]
    assert pair == {"A - 곡1", "B - 곡2"}


async def test_fill_falls_back_to_llm_when_pool_has_no_battle_pair():
    """그 장르 곡 풀이 아예 없으면(예: 재즈) 기존 LLM 취향 질문으로 넘어간다."""
    from app.components.song_pool import SongPool

    repo = FakeRepo()
    generator = FakeGenerator([("질문1", "A1", "B1")])
    pool = SongPool([_pool_song("A", "곡1", "댄스", 50)])  # 재즈 곡 없음

    await w._fill(
        repo,
        generator,
        FakeStatsSource(),
        DAY,
        "재즈",
        set(),
        pool,
        set(),
        random.Random(0),
    )

    assert repo.approved[(DAY, "재즈")] == "질문1"


async def test_fill_does_not_repeat_a_battle_pair_used_recently():
    from app.components.song_pool import SongPool

    repo = FakeRepo()
    generator = FakeGenerator([("취향 질문", "A1", "B1")])
    pool = SongPool([_pool_song("A", "곡1", "댄스", 50), _pool_song("B", "곡2", "댄스", 55)])
    used_pairs = {frozenset({"A - 곡1", "B - 곡2"})}  # 이미 최근에 나온 유일한 짝

    await w._fill(
        repo,
        generator,
        FakeStatsSource(),
        DAY,
        "댄스",
        set(),
        pool,
        used_pairs,
        random.Random(0),
    )

    assert repo.approved[(DAY, "댄스")] == "취향 질문"
