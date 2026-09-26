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

    assert curation.situation_genres == ("댄스",)
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
    assert curation.situation_genres == ()
    assert curation.situation_moods == ()


async def test_curate_with_empty_tags_raises():
    with pytest.raises(LLMError):
        await SongCurator(ScriptedLLM(HAPPY_RESPONSE)).curate_from_tags([])


async def test_curate_passes_situation_text_as_is():
    llm = ScriptedLLM(HAPPY_RESPONSE)
    await SongCurator(llm).curate("상황: 비 오는 저녁, 기온 18도, 장소: 동탄역 카페거리")
    assert llm.last_prompt == "상황: 비 오는 저녁, 기온 18도, 장소: 동탄역 카페거리"


async def test_curate_with_blank_situation_raises():
    with pytest.raises(LLMError):
        await SongCurator(ScriptedLLM(HAPPY_RESPONSE)).curate("   ")


async def test_situation_can_have_several_genres():
    response = (
        "상황: 장르=시티팝,인디,어쿠스틱 | 무드=따뜻한,설레는,잔잔한\n곡:\n"
        "아이유 - 밤편지 | 장르=발라드 | 무드=잔잔한"
    )
    curation = await SongCurator(ScriptedLLM(response)).curate("상황: 맑은 저녁")
    assert curation.situation_genres == ("시티팝", "인디", "어쿠스틱")
    assert curation.situation_moods == ("따뜻한", "설레는", "잔잔한")


async def test_tags_outside_the_vocab_are_dropped():
    """목록 밖 태그는 겹침 계산이 안 되니 버린다."""
    response = (
        "상황: 장르=케이팝,댄스 | 무드=몽글몽글한,신나는\n곡:\n"
        "지코 - 아무노래 | 장르=케이팝 | 무드=신나는"
    )
    curation = await SongCurator(ScriptedLLM(response)).curate("상황: 파티")
    assert curation.situation_genres == ("댄스",)
    assert curation.situation_moods == ("신나는",)
    assert curation.songs[0].genre == ""


async def test_copied_format_example_line_is_ignored():
    """LLM이 출력 형식 예시 줄을 곡처럼 베끼는 경우가 실측에서 있었다."""
    response = (
        "상황: 장르=발라드 | 무드=잔잔한\n곡:\n"
        "<가수> - <곡 제목> | 장르=<장르> | 무드=<무드>,<무드>\n"
        "아이유 - 밤편지 | 장르=발라드 | 무드=잔잔한"
    )
    curation = await SongCurator(ScriptedLLM(response)).curate("상황: 비 오는 밤")
    assert [s.title for s in curation.songs] == ["밤편지"]
