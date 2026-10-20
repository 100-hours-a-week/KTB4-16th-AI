from datetime import UTC, datetime, timedelta

from app.db.repositories.record_sync_repository import EPOCH, ChangedRecord, Cursor
from app.schemas.embeddings import EmbeddingGenerateRequest
from app.workers import record_sync_worker as w

T0 = datetime(2026, 10, 15, 3, 0, tzinfo=UTC)


def _rec(rid, minutes, *, deleted=False, photo="photos/a.jpg", comment="비 오는 날"):
    at = T0 + timedelta(minutes=minutes)
    return ChangedRecord(
        record_id=rid,
        user_id=7,
        photo_key=photo,
        title="밤편지",
        artist_name="아이유",
        external_track_id="3P3UA61WRQqwCXaoFOTENd",
        comment=comment,
        created_at=T0,
        updated_at=at,
        deleted_at=at if deleted else None,
    )


class FakeReader:
    def __init__(self, records):
        self.records = sorted(records, key=lambda r: (r.updated_at, r.record_id))

    async def changed_since(self, cursor, limit):
        after = [
            r
            for r in self.records
            if (r.updated_at, r.record_id) > (cursor.updated_at, cursor.record_id)
        ]
        return after[:limit]


class FakeCursors:
    def __init__(self):
        self.cursor = Cursor(EPOCH, 0)

    async def load(self):
        return self.cursor

    async def save(self, cursor):
        self.cursor = cursor


class FakeJobs:
    def __init__(self):
        self.enqueued, self.cancelled = {}, []

    async def enqueue(self, *, job_type, dedupe_key, payload, max_attempts):
        self.enqueued[dedupe_key] = payload

    async def cancel(self, *, job_type, dedupe_key):
        self.cancelled.append(dedupe_key)


class FakeStore:
    def __init__(self):
        self.deleted = []

    async def delete(self, record_id):
        self.deleted.append(record_id)


class FakeSigner:
    def __init__(self, fail_on=None):
        self.fail_on = fail_on

    def sign(self, key):
        if key == self.fail_on:
            raise RuntimeError("서명 실패")
        return f"https://signed/{key}"


async def _run(records, batch=500, signer=None, cursors=None):
    cursors = cursors or FakeCursors()
    jobs, store = FakeJobs(), FakeStore()
    n = await w.sync(FakeReader(records), cursors, jobs, store, signer or FakeSigner(), batch)
    return n, cursors, jobs, store


async def test_new_record_is_queued_with_original_request_shape():
    n, cursors, jobs, _ = await _run([_rec(1, 1)])
    assert n == 1
    payload = jobs.enqueued["1"]
    req = EmbeddingGenerateRequest.model_validate(payload)  # 기존 임베딩 작업이 그대로 받는다
    assert req.photo_url == "https://signed/photos/a.jpg"
    assert req.track.artist_name == "아이유"
    assert cursors.cursor == Cursor(T0 + timedelta(minutes=1), 1)


async def test_deleted_record_cancels_job_and_removes_vectors():
    _, _, jobs, store = await _run([_rec(2, 1, deleted=True)])
    assert jobs.cancelled == ["2"] and store.deleted == [2]
    assert "2" not in jobs.enqueued


async def test_next_run_reads_only_after_cursor():
    records = [_rec(1, 1), _rec(2, 2)]
    _, cursors, _, _ = await _run(records)
    records.append(_rec(3, 3))
    n, _, jobs, _ = await _run(records, cursors=cursors)
    assert n == 1 and list(jobs.enqueued) == ["3"]


async def test_same_timestamp_records_are_not_skipped_across_batches():
    records = [_rec(i, 1) for i in range(1, 6)]  # 다섯 개가 같은 시각
    n, _, jobs, _ = await _run(records, batch=2)
    assert n == 5 and sorted(jobs.enqueued) == ["1", "2", "3", "4", "5"]


async def test_record_without_photo_is_skipped_but_cursor_moves():
    n, cursors, jobs, _ = await _run([_rec(1, 1, photo=None), _rec(2, 2)])
    assert list(jobs.enqueued) == ["2"] and cursors.cursor.record_id == 2


async def test_photo_signing_failure_skips_only_that_embedding_and_keeps_syncing():
    signer = FakeSigner(fail_on="photos/b.jpg")
    n, cursors, jobs, _ = await _run(
        [_rec(1, 1), _rec(2, 2, photo="photos/b.jpg"), _rec(3, 3)], signer=signer
    )
    assert list(jobs.enqueued) == ["1", "3"]  # 2번은 임베딩만 건너뜀
    assert n == 3 and cursors.cursor.record_id == 3  # 3번까지 동기화는 계속된다


def test_slot_key_groups_five_minutes():
    a = w.slot_key(datetime(2026, 10, 15, 3, 7, 59, tzinfo=UTC))
    b = w.slot_key(datetime(2026, 10, 15, 3, 9, 1, tzinfo=UTC))
    c = w.slot_key(datetime(2026, 10, 15, 3, 10, 0, tzinfo=UTC))
    assert a == b == "2026-10-15T03:05" and c == "2026-10-15T03:10"


def test_gcs_signer_accepts_key_formats():
    from app.clients.photo_signer import GcsPhotoSigner

    s = GcsPhotoSigner("mulo-photos", 120)
    assert s._object_name("gs://mulo-photos/records/1.jpg") == "records/1.jpg"
    assert (
        s._object_name("https://storage.googleapis.com/mulo-photos/records/1.jpg?X=1")
        == "records/1.jpg"
    )
    assert s._object_name("records/1.jpg") == "records/1.jpg"


async def test_schedule_does_nothing_while_sync_is_disabled(monkeypatch):
    from datetime import UTC, datetime

    from app import worker_main
    from app.workers import record_sync_worker

    called = []

    async def fake_schedule(now):
        called.append(now)

    class Off:
        record_sync_enabled = False

    monkeypatch.setattr(record_sync_worker, "schedule", fake_schedule)
    monkeypatch.setattr(worker_main, "_sync_slots", set())
    monkeypatch.setattr(worker_main, "get_settings", lambda: Off())

    await worker_main.schedule_record_sync(datetime(2026, 10, 20, 5, 0, tzinfo=UTC))

    assert called == []


async def test_sync_and_comment_storage_are_on_by_default():
    from app.config import get_settings

    s = get_settings()
    assert s.record_sync_enabled is True and s.record_facts_store_comment is True


class FakeFacts:
    def __init__(self):
        self.rows, self.deleted = {}, []

    async def upsert_many(self, facts):
        for f in facts:
            self.rows[f.record_id] = f

    async def delete_many(self, ids):
        self.deleted.extend(ids)


def _rich(rid, minutes, **extra):
    from dataclasses import replace

    return replace(
        _rec(rid, minutes),
        recorded_at_local=datetime(2026, 10, 15, 21, 30),  # 한국시간 그대로
        weather_condition="RAIN",
        temperature=14.5,
        mood_score=4,
        dong_name="서교동",
        dong_code="1144010100",
        latitude=37.55,
        longitude=126.92,
        **extra,
    )


async def _run_with_facts(records, store_comment=False):
    facts = FakeFacts()
    jobs = FakeJobs()
    await w.sync(
        FakeReader(records),
        FakeCursors(),
        jobs,
        FakeStore(),
        FakeSigner(),
        500,
        facts=facts,
        store_comment=store_comment,
    )
    return facts, jobs


async def test_values_missing_from_the_request_body_are_saved_for_memory_search():
    facts, _ = await _run_with_facts([_rich(1, 1)])

    fact = facts.rows[1]
    assert fact.weather_condition == "RAIN" and fact.temperature == 14.5 and fact.mood_score == 4
    assert fact.dong_name == "서교동" and fact.dong_code == "1144010100"
    assert (fact.latitude, fact.longitude) == (37.55, 126.92)
    assert fact.track_title == "밤편지" and fact.artist_name == "아이유"
    assert fact.recorded_at == datetime(2026, 10, 15, 21, 30)  # 한국시간 그대로, 시간대 없음


async def test_comment_text_is_stored_only_when_enabled():
    off, _ = await _run_with_facts([_rich(1, 1)])
    on, _ = await _run_with_facts([_rich(1, 1)], store_comment=True)

    assert off.rows[1].comment_text is None
    assert on.rows[1].comment_text == "비 오는 날"


async def test_facts_are_saved_even_when_the_record_has_no_photo():
    from dataclasses import replace

    facts, jobs = await _run_with_facts([replace(_rich(1, 1), photo_key=None)])

    assert 1 in facts.rows and jobs.enqueued == {}  # 임베딩은 건너뛰어도 날씨·장소는 남긴다


async def test_deleted_record_is_removed_from_facts():
    facts, _ = await _run_with_facts([_rich(1, 1), _rec(1, 2, deleted=True)])

    assert facts.deleted == [1]


async def test_record_facts_are_saved_even_when_photo_signing_fails():
    from dataclasses import replace

    facts = FakeFacts()
    bad = replace(_rich(1, 1), photo_key="photos/b.jpg")
    await w.sync(
        FakeReader([bad]),
        FakeCursors(),
        FakeJobs(),
        FakeStore(),
        FakeSigner(fail_on="photos/b.jpg"),
        500,
        facts=facts,
    )
    assert 1 in facts.rows  # 원문은 사진 문제와 상관없이 저장된다


def test_jsonable_turns_database_values_into_json_safe_ones():
    from datetime import date, time
    from decimal import Decimal

    from app.db.repositories.record_sync_repository import jsonable

    row = {
        "created_at": datetime(2026, 10, 15, 21, 30),
        "day": date(2026, 10, 15),
        "t": time(21, 30),
        "temp": Decimal("14.5"),
        "blob": b"\x01\xff",
        "n": None,
        "nested": [Decimal("1.5"), {"x": datetime(2026, 1, 1)}],
    }

    assert jsonable(row) == {
        "created_at": "2026-10-15T21:30:00",
        "day": "2026-10-15",
        "t": "21:30:00",
        "temp": 14.5,
        "blob": "01ff",
        "n": None,
        "nested": [1.5, {"x": "2026-01-01T00:00:00"}],
    }


async def test_whole_record_is_kept_as_raw_json_in_the_facts():
    from dataclasses import replace

    raw = {"record_id": 1, "visibility": "PRIVATE", "comment": "비 오는 날", "extra_col": 7}
    facts, _ = await _run_with_facts([replace(_rich(1, 1), raw_record=raw)], store_comment=True)

    assert facts.rows[1].raw_record == raw  # 코드가 모르는 칸(extra_col)도 그대로 남는다


async def test_raw_json_drops_the_comment_when_comment_storage_is_off():
    from dataclasses import replace

    raw = {"record_id": 1, "comment": "비 오는 날", "extra_col": 7}
    facts, _ = await _run_with_facts([replace(_rich(1, 1), raw_record=raw)], store_comment=False)

    assert facts.rows[1].raw_record == {"record_id": 1, "extra_col": 7}


class _CapturingMysql:
    """백엔드 DB 대신 보낸 SQL만 기록하고 빈 결과를 돌려준다."""

    def __init__(self):
        self.statements = []

    async def execute(self, statement, params=None):
        self.statements.append(" ".join(str(statement).split()))
        return []


async def test_changed_time_falls_back_to_created_at_when_updated_at_is_null():
    from app.db.repositories.record_sync_repository import BackendRecordReader

    mysql = _CapturingMysql()
    await BackendRecordReader(mysql).changed_since(Cursor(EPOCH, 0), 10)

    # 수정한 적 없는 자물쇠는 updated_at이 NULL이다. GREATEST는 NULL이 하나라도 있으면 NULL이라
    # 칸마다 created_at으로 대신하지 않으면 그런 자물쇠가 한 건도 읽히지 않는다
    assert (
        "GREATEST(COALESCE(r.updated_at, r.created_at), COALESCE(r.deleted_at, r.created_at))"
        in mysql.statements[0]
    )
