import pytest

from app.components.axis_vectors import DEFAULT_AXES_PATH, load_axis_specs
from app.components.song_curator import (
    EXPRESSION_AXIS,
    KIND_AXIS,
    MEME_KIND,
    SongCurator,
    photo_situation,
)
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
        "지코 - 아무노래 | 장르=랩/힙합 | 무드=신나는"
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
        "상황: 장르=POP,인디음악,포크/블루스 | 무드=따뜻한,설레는,잔잔한\n곡:\n"
        "아이유 - 밤편지 | 장르=발라드 | 무드=잔잔한"
    )
    curation = await SongCurator(ScriptedLLM(response)).curate("상황: 맑은 저녁")
    assert curation.situation_genres == ("POP", "인디음악", "포크/블루스")
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


async def test_old_genre_names_are_dropped():
    """2026-10-08 장르를 13개로 바꾼 뒤 예전 이름(시티팝·힙합 등)은 목록 밖이라 버린다."""
    response = (
        "상황: 장르=시티팝,힙합,랩/힙합 | 무드=신나는\n곡:\n"
        "지코 - 아무노래 | 장르=힙합 | 무드=신나는"
    )
    curation = await SongCurator(ScriptedLLM(response)).curate("상황: 파티")
    assert curation.situation_genres == ("랩/힙합",)
    assert curation.songs[0].genre == ""


def test_situation_without_axes_is_the_old_sentence():
    assert photo_situation(["카페", "실내"]) == "사진에서 뽑은 분위기 태그: 카페, 실내"
    assert photo_situation(["카페"], {}) == "사진에서 뽑은 분위기 태그: 카페"


def test_situation_of_a_real_photo_lists_kind_scene_and_other_axes():
    axes = {"사진 종류": "진짜 사진", "대상": "여러 사람", "분위기": "밤의 따뜻한 조명"}

    assert photo_situation(["카페"], axes) == (
        "사진에서 뽑은 정보 — 사진 종류: 진짜 사진 / 장소 태그: 카페 / "
        "대상: 여러 사람 / 분위기: 밤의 따뜻한 조명"
    )


def test_situation_of_a_meme_with_expression_drops_the_meaningless_scene_tag():
    axes = {"사진 종류": "그림·짤", "표정·감정": "슬픔·눈물"}

    situation = photo_situation(["콘서트"], axes)

    assert situation == "사진에서 뽑은 정보 — 사진 종류: 그림·짤 / 표정·감정: 슬픔·눈물"
    assert "콘서트" not in situation


def test_situation_of_a_meme_without_expression_keeps_the_scene_tag():
    """표정이 불확실하면 알려 줄 게 장소 태그뿐이라 그대로 보낸다."""
    assert photo_situation(["콘서트"], {"사진 종류": "그림·짤"}) == (
        "사진에서 뽑은 정보 — 사진 종류: 그림·짤 / 장소 태그: 콘서트"
    )


async def test_curate_from_tags_sends_the_axes_to_the_llm():
    llm = ScriptedLLM(HAPPY_RESPONSE)

    await SongCurator(llm).curate_from_tags(
        ["카페"], axes={"사진 종류": "진짜 사진", "대상": "음식·음료"}
    )

    assert llm.last_prompt == (
        "사진에서 뽑은 정보 — 사진 종류: 진짜 사진 / 장소 태그: 카페 / 대상: 음식·음료"
    )


def test_axis_names_used_in_the_situation_exist_in_the_axes_file():
    """song_curator의 이름 상수가 axis_tags.yaml과 달라지면 그림·짤 규칙이 조용히 안 먹는다."""
    specs = {s.ko: s for s in load_axis_specs(DEFAULT_AXES_PATH)}

    assert KIND_AXIS in specs
    assert MEME_KIND in {t.ko for t in specs[KIND_AXIS].tags}
    assert EXPRESSION_AXIS in specs
    assert specs[EXPRESSION_AXIS].when == MEME_KIND


async def test_curate_reads_the_mood_sentence_line():
    response = HAPPY_RESPONSE.replace(
        "곡:", "분위기: 퇴근길 창밖 빗소리처럼 조용하고 쓸쓸한 노래\n곡:", 1
    )

    curation = await SongCurator(ScriptedLLM(response)).curate_from_tags(["카페"])

    assert curation.mood_sentence == "퇴근길 창밖 빗소리처럼 조용하고 쓸쓸한 노래"
    assert len(curation.songs) == 2  # 분위기 줄이 곡 줄로 읽히지 않는다


async def test_curate_without_mood_sentence_or_with_copied_template_gives_empty():
    plain = await SongCurator(ScriptedLLM(HAPPY_RESPONSE)).curate_from_tags(["카페"])
    copied = HAPPY_RESPONSE.replace("곡:", "분위기: <한 문장>\n곡:", 1)
    template = await SongCurator(ScriptedLLM(copied)).curate_from_tags(["카페"])

    assert plain.mood_sentence == "" and template.mood_sentence == ""
