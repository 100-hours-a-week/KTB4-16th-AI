"""기억 검색이 자물쇠 조건 값을 어디서 읽을지 정하는 설정 (MEMORY_RECORD_SOURCE)."""

from types import SimpleNamespace

import pytest
from sqlalchemy.exc import SQLAlchemyError

import app.dependencies as deps
from app.exceptions import UpstreamError


def _settings(monkeypatch, **values):
    base = {"ai_client_mode": "real", "memory_record_source": "mysql", "backend_mysql_url": ""}
    base.update(values)
    monkeypatch.setattr(deps, "get_settings", lambda: SimpleNamespace(**base))


def test_default_reads_the_backend_mysql(monkeypatch):
    _settings(monkeypatch, backend_mysql_url="mysql+aiomysql://x")

    assert isinstance(deps.get_memory_record_source(), deps._DbMemoryRecords)


def test_without_a_backend_url_and_on_mysql_nothing_can_be_read(monkeypatch):
    _settings(monkeypatch)

    assert isinstance(deps.get_memory_record_source(), deps._NoMemoryRecords)


def test_postgres_source_needs_no_backend_url(monkeypatch):
    _settings(monkeypatch, memory_record_source="postgres")  # 백엔드 DB 주소가 없다

    assert isinstance(deps.get_memory_record_source(), deps._PgMemoryRecords)


def test_fake_mode_never_reads_a_database_even_on_postgres(monkeypatch):
    _settings(monkeypatch, ai_client_mode="fake", memory_record_source="postgres")

    assert isinstance(deps.get_memory_record_source(), deps._NoMemoryRecords)


@pytest.mark.parametrize(
    "call",
    [
        lambda s: s.existing_ids(user_id=7, record_ids=[1]),
        lambda s: s.list_condition_rows(user_id=7, limit=10),
        lambda s: s.list_track_names(user_id=7, limit=10),
    ],
)
async def test_postgres_read_failure_becomes_an_upstream_error(monkeypatch, call):
    def broken():
        raise SQLAlchemyError("연결 실패")

    monkeypatch.setattr(deps.postgres, "get_sessionmaker", broken)

    with pytest.raises(UpstreamError):  # 오류 계약대로 503이 된다
        await call(deps._PgMemoryRecords())
