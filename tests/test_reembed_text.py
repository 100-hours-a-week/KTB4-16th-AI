"""재임베딩 도구(app/tools/reembed_text.py)의 순수 계산 부분.

DB를 읽고 쓰는 부분은 pgvector DB가 없으면 실행할 수 없어서, 어떤 값으로 바꾸는지만 확인한다.
"""

from app.components.music_mood_tagger import PROMPT_VERSION
from app.components.track_mood_resolver import build_mood_version
from app.tools.reembed_text import next_mood_version, next_record_versions

LLM = "claude-test"
OLD = build_mood_version(LLM, "text-embedding-3-small")


def test_mood_version_swaps_only_the_embedding_model():
    new = next_mood_version(OLD, LLM, "nlpai-lab/KURE-v1")

    assert new == f"{LLM}|{PROMPT_VERSION}|nlpai-lab/KURE-v1"
    assert new == build_mood_version(LLM, "nlpai-lab/KURE-v1")  # 서비스가 찾는 값과 같다


def test_mood_made_by_another_llm_or_prompt_keeps_its_old_version():
    """지금 설정으로는 어차피 다시 만들어질 행이다 — 버전을 올려서 잘못 맞는 캐시가 되면 안 된다."""
    other_llm = build_mood_version("other-llm", "text-embedding-3-small")
    other_prompt = f"{LLM}|music-mood-v0|text-embedding-3-small"

    assert next_mood_version(other_llm, LLM, "kure") == other_llm
    assert next_mood_version(other_prompt, LLM, "kure") == other_prompt


def test_record_versions_keep_other_fields_such_as_tag_version():
    old = {"image": "openai/clip", "text": "text-embedding-3-small", "tagVersion": "tag-v1"}

    new = next_record_versions(old, "nlpai-lab/KURE-v1", "mv")

    assert new == {
        "image": "openai/clip",
        "text": "nlpai-lab/KURE-v1",
        "tagVersion": "tag-v1",
        "moodVersion": "mv",
    }
    assert old["text"] == "text-embedding-3-small"  # 원본은 그대로


def test_record_versions_handle_missing_old_value():
    assert next_record_versions(None, "m", "mv") == {"text": "m", "moodVersion": "mv"}


async def test_track_mood_cache_ignores_rows_not_re_embedded_yet():
    """0010 뒤 재임베딩 전의 곡 무드(벡터 NULL)는 캐시 적중으로 돌려주면 안 된다 — 새로 만든다."""
    from sqlalchemy.dialects import postgresql

    from app.db.repositories.track_mood_repository import TrackMoodRepository

    class CapturingSession:
        def __init__(self):
            self.statements = []

        async def execute(self, statement):
            self.statements.append(statement)

            class _Result:
                def first(self):
                    return None

            return _Result()

    session = CapturingSession()

    assert await TrackMoodRepository(session).get("t1", "v") is None

    sql = str(session.statements[0].compile(dialect=postgresql.dialect()))
    assert "track_moods.mood_embedding IS NOT NULL" in sql
