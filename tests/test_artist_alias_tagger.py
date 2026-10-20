"""가수 이름 → 다른 글자들(표기·철자·오타·별명) 변환기 — LLM 응답 다듬기."""

import pytest

from app.components.artist_alias_tagger import (
    MAX_ALIASES,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    ArtistAliasTagger,
    needs_alias,
)


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


async def _aliases(name: str, reply: str) -> tuple[str, ...]:
    return await ArtistAliasTagger(StubLLM(reply)).aliases(name)


async def test_the_artist_name_is_sent_with_a_prompt_that_asks_for_every_kind_of_spelling():
    llm = StubLLM("표기: 빅뱅")

    assert await ArtistAliasTagger(llm).aliases("BIGBANG") == ("빅뱅",)
    system, prompt = llm.calls[0]
    assert prompt == "가수: BIGBANG" and system == SYSTEM_PROMPT
    # 다른 언어 표기, 다른 철자, 오타, 줄임말·별명, 모르는 가수는 발음대로
    for asked in ("영어↔한글", "철자", "오타", "별명", "발음"):
        assert asked in system


async def test_korean_name_gets_english_spellings_too():
    assert await _aliases("아이유", "표기: IU, 이지은, Lee Jieun") == ("IU", "이지은", "Lee Jieun")


async def test_nicknames_real_names_and_spelling_variants_are_kept():
    reply = "표기: 지디, 지드래곤, GD, 권지용"

    assert await _aliases("G-DRAGON", reply) == ("지디", "지드래곤", "GD", "권지용")
    assert await _aliases("Billie Eilish", "표기: 빌리 아일리시, 빌리 아일리쉬") == (
        "빌리 아일리시",
        "빌리 아일리쉬",
    )


async def test_a_name_that_is_not_an_artist_gives_an_empty_result_not_an_error():
    assert await _aliases("Various Artists", "표기: 없음") == ()


async def test_duplicates_and_spelling_only_differences_are_kept_once():
    # "빅 뱅"은 공백만 다른 같은 글자라 한 번만
    assert await _aliases("BIGBANG", "표기: 빅뱅, 빅 뱅, 뱅뱅, 빅뱅") == ("빅뱅", "뱅뱅")


async def test_the_stored_name_itself_is_not_an_alias():
    assert await _aliases("RESCENE", "표기: RESCENE, Rescene, Res Cene, rescene") == ()


async def test_at_most_six_aliases_are_kept_in_the_given_order():
    reply = "표기: " + ", ".join(f"별명{n}" for n in range(1, 10))

    found = await _aliases("BTS", reply)

    assert len(found) == MAX_ALIASES == 6 and found[0] == "별명1" and found[-1] == "별명6"


@pytest.mark.parametrize(
    ("name", "reply", "expected"),
    [
        ("BIGBANG", "표기: 빅뱅, 빅뱅그룹, 빅뱅 멤버, 빅뱅 노래", ("빅뱅",)),
        ("잔나비", "표기: Jannabi, 잔나비밴드, Jannabi band", ("Jannabi",)),
        ("아이유", "표기: IU, 아이유님, IU singer", ("IU",)),
        # 이름 자체에 들어 있는 말이면 거르지 않는다
        ("더 밴드 페리", "표기: 밴드 페리, The Band Perry", ("밴드 페리", "The Band Perry")),
    ],
)
async def test_answers_with_filler_words_are_dropped_unless_the_name_has_them(
    name, reply, expected
):
    assert await _aliases(name, reply) == expected


async def test_too_short_answers_and_the_none_marker_are_dropped():
    assert await _aliases("BIGBANG", "표기: 가, 없음, ,  ") == ()


async def test_text_around_the_answer_line_is_ignored():
    reply = "알겠습니다.\n표기: 지드래곤\n(참고: 솔로 활동명)"

    assert await _aliases("G-DRAGON", reply) == ("지드래곤",)


async def test_an_answer_without_the_expected_line_is_an_error_so_it_is_not_saved():
    with pytest.raises(ValueError):
        await _aliases("BIGBANG", "잘 모르겠어요")


async def test_llm_failure_is_raised_to_the_caller():
    with pytest.raises(RuntimeError):
        await ArtistAliasTagger(StubLLM(error=RuntimeError("boom"))).aliases("BIGBANG")


def test_version_changes_with_the_model_and_the_prompt():
    assert ArtistAliasTagger(StubLLM()).version == f"stub-llm|{PROMPT_VERSION}"
    assert PROMPT_VERSION == "artist-alias-v2"  # 프롬프트를 바꾸면 이미 만든 별칭을 다시 만든다


@pytest.mark.parametrize(("name", "expected"), [("-", False), ("...", False), ("10CM", True)])
def test_names_without_any_letter_have_nothing_to_ask(name, expected):
    assert needs_alias(name) is expected
