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


def test_small_group_and_unknown_genre_go_to_common():
    saved = [_s(1, "b1"), _s(2, "b1"), _s(3, "i1"), _s(4, "old"), _s(5, "old2")]
    result = assign_genres(saved, GENRES, min_group_size=2)
    assert result[1].music_genre == "발라드" and result[2].music_genre == "발라드"
    assert result[3].music_genre == "공통"  # 인디 1명뿐 → 공통
    assert result[4].music_genre == "공통" and result[4].source == "common"  # 13개 밖 장르


def test_small_common_group_merges_into_biggest_genre():
    # 발라드 3명, 인디 1명(→공통), 장르 모름 1명(→공통) / 기준 3명 → 공통 2명 < 3 → 발라드로 합침
    saved = [_s(1, "b1"), _s(2, "b1"), _s(3, "b2"), _s(4, "i1"), _s(5, "old")]
    result = assign_genres(saved, GENRES, min_group_size=3)
    assert {result[u].music_genre for u in (1, 2, 3, 4, 5)} == {"발라드"}
    assert result[4].source == "merged" and result[1].source == "records"


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
        self.spares: list[dict] = list(spares)

    async def add_question(
        self, *, question_date, music_genre, question, option_a, option_b, model_version
    ):
        key = (question_date, music_genre)
        if key in self.approved:
            return False
        self.approved[key] = question
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

    await w._fill(repo, generator, FakeStatsSource(), DAY, "발라드", used)

    assert repo.approved[(DAY, "발라드")] == "질문1"
    assert [s["question"] for s in repo.spares] == ["질문2", "질문3"]
    assert used == {"질문1", "질문2", "질문3"}


async def test_fill_skips_candidates_already_used_recently():
    repo = FakeRepo()
    generator = FakeGenerator([("옛 질문", "A1", "B1"), ("새 질문", "A2", "B2")])
    used = {"옛 질문"}

    await w._fill(repo, generator, FakeStatsSource(), DAY, "댄스", used)

    assert repo.approved[(DAY, "댄스")] == "새 질문"


async def test_fill_uses_a_banked_spare_when_no_fresh_candidate():
    """새 후보가 다 최근 질문과 겹치면 LLM을 또 안 부르고 쌓아 둔 예비 질문을 쓴다."""
    repo = FakeRepo(spares=[{"genre": "인디음악", "question": "예비 질문"}])
    generator = FakeGenerator([("옛 질문", "A", "B")])
    used = {"옛 질문"}

    await w._fill(repo, generator, FakeStatsSource(), DAY, "인디음악", used)

    assert repo.approved[(DAY, "인디음악")] == "예비 질문"
    assert repo.spares == []  # 한 번 쓴 예비 질문은 다시 안 쓴다
    assert "예비 질문" in used


async def test_fill_raises_when_no_fresh_candidate_and_no_spare():
    repo = FakeRepo()
    generator = FakeGenerator([("옛 질문", "A", "B")])
    used = {"옛 질문"}

    with pytest.raises(GroupFailedError):
        await w._fill(repo, generator, FakeStatsSource(), DAY, "재즈", used)


async def test_fill_does_not_reuse_a_spare_from_a_different_genre():
    repo = FakeRepo(spares=[{"genre": "트로트", "question": "트로트 예비"}])
    generator = FakeGenerator([("옛 질문", "A", "B")])
    used = {"옛 질문"}

    with pytest.raises(GroupFailedError):
        await w._fill(repo, generator, FakeStatsSource(), DAY, "재즈", used)
    assert repo.spares  # 다른 장르 예비는 그대로 남아 있다
