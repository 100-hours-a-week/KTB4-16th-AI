from datetime import UTC, datetime, timedelta

import pytest

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


async def test_failure_keeps_cursor_at_last_success():
    signer = FakeSigner(fail_on="photos/b.jpg")
    cursors = FakeCursors()
    with pytest.raises(RuntimeError):
        await _run([_rec(1, 1), _rec(2, 2, photo="photos/b.jpg")], signer=signer, cursors=cursors)
    assert cursors.cursor.record_id == 1  # 다음 실행에서 2번부터 다시


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
    from app.config import get_settings
    from app.workers import record_sync_worker

    called = []

    async def fake_schedule(now):
        called.append(now)

    monkeypatch.setattr(record_sync_worker, "schedule", fake_schedule)
    monkeypatch.setattr(worker_main, "_sync_slots", set())
    assert get_settings().record_sync_enabled is False  # 기본은 꺼짐

    await worker_main.schedule_record_sync(datetime(2026, 10, 20, 5, 0, tzinfo=UTC))

    assert called == []


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
