"""가수 이름 → 다른 표기(영어↔한글) 변환기 — 방향 고르기와 LLM 응답 다듬기."""

import pytest

from app.components.artist_alias_tagger import PROMPT_VERSION, ArtistAliasTagger, needs_alias


class StubLLM:
    model = "stub-llm"

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, str]] = []

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.calls.append((system, prompt))
        if self.error:
            raise self.error
        return self.reply


async def test_english_name_asks_for_the_korean_spelling():
    llm = StubLLM("표기: 빅뱅")

    assert await ArtistAliasTagger(llm).aliases("BIGBANG") == ("빅뱅",)
    system, prompt = llm.calls[0]
    assert "한글 표기" in system and prompt == "가수: BIGBANG"


async def test_korean_name_asks_for_the_english_spelling():
    llm = StubLLM("표기: IU")

    assert await ArtistAliasTagger(llm).aliases("아이유") == ("IU",)
    assert "영어(로마자) 표기" in llm.calls[0][0]


async def test_an_unknown_artist_gives_an_empty_result_not_an_error():
    assert await ArtistAliasTagger(StubLLM("표기: 없음")).aliases("SunLake") == ()


async def test_several_spellings_are_kept_without_duplicates_and_capped_at_three():
    llm = StubLLM("표기: 빅뱅, 빅 뱅, 뱅뱅, 팡팡, 퐁퐁")

    # "빅 뱅"은 공백만 다른 같은 이름이라 뺀다
    assert await ArtistAliasTagger(llm).aliases("BIGBANG") == ("빅뱅", "뱅뱅", "팡팡")


async def test_text_around_the_answer_line_is_ignored():
    llm = StubLLM("알겠습니다.\n표기: 지드래곤\n(참고: 솔로 활동명)")

    assert await ArtistAliasTagger(llm).aliases("G-DRAGON") == ("지드래곤",)


@pytest.mark.parametrize(
    ("name", "reply"),
    [
        ("RESCENE", "표기: RESCENE, Rescene"),  # 한글이 아닌 답은 버린다
        ("아이유", "표기: 아이유, 이지은"),  # 영어가 아닌 답은 버린다
        ("RESCENE", "표기: 없음, "),
    ],
)
async def test_answers_in_the_same_script_as_the_stored_name_are_dropped(name, reply):
    assert await ArtistAliasTagger(StubLLM(reply)).aliases(name) == ()


async def test_an_answer_without_the_expected_line_is_an_error_so_it_is_not_saved():
    with pytest.raises(ValueError):
        await ArtistAliasTagger(StubLLM("잘 모르겠어요")).aliases("BIGBANG")


async def test_llm_failure_is_raised_to_the_caller():
    with pytest.raises(RuntimeError):
        await ArtistAliasTagger(StubLLM(error=RuntimeError("boom"))).aliases("BIGBANG")


def test_version_changes_with_the_model_and_the_prompt():
    assert ArtistAliasTagger(StubLLM()).version == f"stub-llm|{PROMPT_VERSION}"


@pytest.mark.parametrize(("name", "expected"), [("-", False), ("...", False), ("10CM", True)])
def test_names_without_any_letter_have_nothing_to_ask(name, expected):
    assert needs_alias(name) is expected
