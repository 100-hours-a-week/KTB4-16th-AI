from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace

from app.tools.backfill_record_facts import backfill, fact_from_row


def _row(record_id=1, **overrides):
    values = {
        "record_id": record_id,
        "user_id": 7,
        "created_at": datetime(2026, 10, 15, 21, 30),
        "weather_condition": "CLOUDY",
        "temperature": Decimal("14.5"),
        "mood_score": -10,
        "comment": "  광교산 단풍  ",
        "legal_dong_name": "성수동1가",
        "legal_dong_code": 1120011500,
        "latitude": Decimal("37.544000"),
        "longitude": Decimal("127.037000"),
        "track_title": "소나기",
        "artist_name": "RESCENE, 리센느",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_fact_converts_decimals_and_numeric_codes_and_keeps_the_time_as_is():
    fact = fact_from_row(_row(), store_comment=False)

    assert fact.temperature == 14.5 and isinstance(fact.temperature, float)
    assert (fact.latitude, fact.longitude) == (37.544, 127.037)
    assert fact.dong_code == "1120011500"  # 숫자로 와도 문자열로
    assert fact.recorded_at == datetime(2026, 10, 15, 21, 30)  # 한국시간 그대로
    assert (fact.track_title, fact.artist_name) == ("소나기", "RESCENE, 리센느")


def test_comment_is_not_stored_unless_asked_and_is_trimmed_when_stored():
    assert fact_from_row(_row(), store_comment=False).comment_text is None
    assert fact_from_row(_row(), store_comment=True).comment_text == "광교산 단풍"
    assert fact_from_row(_row(comment="   "), store_comment=True).comment_text is None
    assert fact_from_row(_row(comment=None), store_comment=True).comment_text is None


def test_missing_values_stay_none_so_records_without_a_place_or_track_are_not_lost():
    fact = fact_from_row(
        _row(
            weather_condition=None,
            temperature=None,
            mood_score=None,
            legal_dong_name=None,
            legal_dong_code=None,
            latitude=None,
            longitude=None,
            track_title=None,
            artist_name=None,
        ),
        store_comment=False,
    )

    assert fact.weather_condition is None and fact.temperature is None and fact.mood_score is None
    assert fact.dong_name is None and fact.dong_code is None
    assert fact.latitude is None and fact.longitude is None
    assert fact.track_title is None and fact.artist_name is None


# ---- 흐름 (가짜 MySQL·PostgreSQL) ----------------------------------------------


class _Rows(list):
    def all(self):
        return list(self)


class _FakeMysql:
    def __init__(self, rows):
        self.rows = rows
        self.params: list[dict] = []
        self.sql = ""

    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def execute(self, statement, params):
        self.sql = str(statement)
        self.params.append(params)
        after = [r for r in self.rows if r.record_id > params["after_id"]]
        return _Rows(after[: params["limit"]])


class _FakePg:
    def __call__(self):
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeRepository:
    saved: list = []

    def __init__(self, session):
        pass

    async def upsert_many(self, facts):
        _FakeRepository.saved.extend(facts)
        return len(facts)


async def test_backfill_reads_in_record_id_order_in_batches_and_saves_everything():
    _FakeRepository.saved = []
    mysql = _FakeMysql([_row(i) for i in (1, 2, 3, 4, 5)])

    counts = await backfill(
        mysql,
        _FakePg(),
        batch_size=2,
        dry_run=False,
        store_comment=False,
        repository=_FakeRepository,
    )

    assert counts.records == 5
    assert [f.record_id for f in _FakeRepository.saved] == [1, 2, 3, 4, 5]
    assert [p["after_id"] for p in mysql.params] == [0, 2, 4, 5]  # 이어서 읽는다
    assert "r.deleted_at IS NULL" in mysql.sql  # 삭제된 자물쇠는 옮기지 않는다
    assert (
        "LEFT JOIN places" in mysql.sql and "LEFT JOIN music_tracks" in mysql.sql
    )  # 빠뜨리지 않는다
    assert "album_image_url" not in mysql.sql and "external_url" not in mysql.sql


async def test_dry_run_counts_but_saves_nothing():
    _FakeRepository.saved = []
    mysql = _FakeMysql([_row(i, comment="있음") for i in (1, 2)] + [_row(3, comment=None)])

    counts = await backfill(
        mysql,
        _FakePg(),
        batch_size=10,
        dry_run=True,
        store_comment=True,
        repository=_FakeRepository,
    )

    assert counts.records == 3 and counts.with_comment == 2
    assert _FakeRepository.saved == []


async def test_backfill_of_an_empty_backend_finishes_with_zero():
    counts = await backfill(
        _FakeMysql([]), _FakePg(), batch_size=10, dry_run=False, store_comment=False
    )

    assert counts.records == 0
