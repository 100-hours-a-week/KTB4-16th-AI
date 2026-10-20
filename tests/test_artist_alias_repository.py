"""artist_aliases 저장소 — 어떤 표를 어떤 조건으로 읽고 쓰는지 생성되는 SQL을 확인한다.

로컬·CI에 pgvector DB가 없어서 실제로 실행하지는 못한다 (실제 DB 검증은 따로 했다).
"""

from types import SimpleNamespace

from sqlalchemy.dialects import postgresql

from app.db.repositories.artist_alias_repository import ArtistAliasRepository


class CapturingSession:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.statements = []
        self.commits = 0

    async def execute(self, statement):
        self.statements.append(statement)
        return list(self.rows)

    async def commit(self):
        self.commits += 1


def _compile(statement):
    compiled = statement.compile(dialect=postgresql.dialect())
    return " ".join(str(compiled).split()), compiled.params


async def test_aliases_are_read_by_name_key_and_empty_lists_are_left_out():
    session = CapturingSession([SimpleNamespace(name_key="rescene", aliases=["리센느"])])

    found = await ArtistAliasRepository(session).aliases_for({"rescene", "iu"})

    sql, _ = _compile(session.statements[0])
    assert "FROM artist_aliases" in sql and "artist_aliases.name_key IN" in sql
    assert "cardinality(artist_aliases.aliases) > %(cardinality_1)s" in sql
    assert found == {"rescene": ["리센느"]}


async def test_reading_aliases_for_no_artists_does_not_touch_the_database():
    session = CapturingSession()

    assert await ArtistAliasRepository(session).aliases_for(set()) == {}
    assert session.statements == []


async def test_source_names_are_the_distinct_artist_columns_of_record_facts():
    session = CapturingSession([SimpleNamespace(artist_name="BIGBANG")])

    names = await ArtistAliasRepository(session).source_artist_names(100)

    sql, _ = _compile(session.statements[0])
    assert "SELECT DISTINCT record_facts.artist_name FROM record_facts" in sql
    assert "record_facts.artist_name IS NOT NULL" in sql
    assert names == ["BIGBANG"]


async def test_known_keys_are_the_given_version_and_the_ones_a_person_entered():
    session = CapturingSession([SimpleNamespace(name_key="bigbang")])

    keys = await ArtistAliasRepository(session).known_keys("model|v1")

    sql, params = _compile(session.statements[0])
    assert "artist_aliases.alias_version IN" in sql
    # 사람이 고쳐 넣은 행은 다시 변환해서 덮어쓰지 않는다
    assert [["model|v1", "manual"]] == [v for v in params.values() if isinstance(v, list)]
    assert keys == {"bigbang"}


async def test_save_overwrites_an_existing_artist_and_keeps_empty_results():
    session = CapturingSession()

    await ArtistAliasRepository(session).save(
        name_key="sunlake", artist_name="SunLake", aliases=[], version="model|v1"
    )

    sql, params = _compile(session.statements[0])
    assert sql.startswith("INSERT INTO artist_aliases")
    assert "ON CONFLICT (name_key) DO UPDATE SET" in sql and "updated_at = now()" in sql
    assert params["aliases"] == [] and params["name_key"] == "sunlake"
    assert session.commits == 1


async def test_save_cuts_a_too_long_artist_name_to_the_column_size():
    session = CapturingSession()

    await ArtistAliasRepository(session).save(
        name_key="k", artist_name="가" * 300, aliases=["a"], version="v"
    )

    _, params = _compile(session.statements[0])
    assert len(params["artist_name"]) == 200
