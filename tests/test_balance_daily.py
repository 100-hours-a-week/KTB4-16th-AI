from datetime import UTC, date, datetime

from app.components.music_genre_assigner import SavedTrack, assign_genres
from app.workers import balance_daily_worker as w
from tests.conftest import AUTH

T = datetime(2026, 10, 1, tzinfo=UTC)
LATER = datetime(2026, 10, 5, tzinfo=UTC)


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
