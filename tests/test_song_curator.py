import pytest

from app.components.song_curator import SongCurator
from app.exceptions import LLMError


class ScriptedLLM:
    model = "scripted-llm"

    def __init__(self, response: str):
        self._response = response
        self.last_prompt: str | None = None

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.last_prompt = prompt
        return self._response


HAPPY_RESPONSE = """상황: 장르=댄스 | 무드=신나는,화려한
곡:
아이유 - 밤편지 | 장르=발라드 | 무드=잔잔한,그리운
The Weeknd - Blinding Lights | 장르=댄스 | 무드=화려한,신나는"""


async def test_curate_parses_situation_and_songs():
    curation = await SongCurator(ScriptedLLM(HAPPY_RESPONSE)).curate_from_tags(["카페", "자연"])

    assert curation.situation_genre == "댄스"
    assert curation.situation_moods == ("신나는", "화려한")
    assert [(s.artist, s.title, s.genre, s.moods) for s in curation.songs] == [
        ("아이유", "밤편지", "발라드", ("잔잔한", "그리운")),
        ("The Weeknd", "Blinding Lights", "댄스", ("화려한", "신나는")),
    ]


async def test_search_query_pins_both_title_and_artist_with_field_filters():
    curation = await SongCurator(ScriptedLLM(HAPPY_RESPONSE)).curate_from_tags(["카페"])
    assert curation.songs[0].search_query == "track:밤편지 artist:아이유"


async def test_curate_skips_unparseable_song_lines_but_keeps_the_rest():
    response = (
        "상황: 장르=댄스 | 무드=신나는\n"
        "곡:\n"
        "아이유 - 밤편지 | 장르=발라드 | 무드=잔잔한\n"
        "이 줄은 형식이 깨짐\n"
        "지코 - 아무노래 | 장르=힙합 | 무드=신나는"
    )
    curation = await SongCurator(ScriptedLLM(response)).curate_from_tags(["카페"])
    assert [s.artist for s in curation.songs] == ["아이유", "지코"]


async def test_curate_returns_empty_songs_when_nothing_parseable():
    curation = await SongCurator(ScriptedLLM("추천할 곡이 없습니다")).curate_from_tags(["카페"])
    assert curation.songs == []
    assert curation.situation_genre == ""
    assert curation.situation_moods == ()


async def test_curate_with_empty_tags_raises():
    with pytest.raises(LLMError):
        await SongCurator(ScriptedLLM(HAPPY_RESPONSE)).curate_from_tags([])
