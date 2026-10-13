import pytest

from app.components.answer_headline import (
    MAX_LENGTH,
    SYSTEM_PROMPT,
    HeadlineWriter,
    build_prompt,
    is_valid,
)
from tests.memory_fakes import StubLLM

# ---- LLM에 보내는 글 ---------------------------------------------------------


def test_prompt_has_only_condition_names_and_counts():
    prompt = build_prompt(["홍대 근처", "비 오는 날"], found=2, total=2, suggestions=0)

    assert prompt == "질문한 조건: 홍대 근처, 비 오는 날\n조건에 모두 맞는 자물쇠: 2건"


def test_prompt_says_how_many_are_shown_when_there_are_more():
    prompt = build_prompt(["가을"], found=50, total=120, suggestions=0)

    assert prompt.splitlines()[1] == "조건에 모두 맞는 자물쇠: 총 120건 (그중 50건을 보여줌)"


def test_prompt_tells_when_nothing_matches_all_conditions_but_some_match_in_part():
    prompt = build_prompt(["가을", "서울숲 근처"], found=0, total=0, suggestions=3)

    assert prompt.splitlines()[1:] == [
        "조건에 모두 맞는 자물쇠: 0건 (없음)",
        "일부만 맞는 자물쇠: 3건 (문장 아래에 따로 안내됨)",
    ]


def test_system_prompt_forbids_inventing_facts_and_listing_records():
    assert "지어내지 않는다" in SYSTEM_PROMPT
    assert "나열하지 않는다" in SYSTEM_PROMPT
    assert "아라비아 숫자" in SYSTEM_PROMPT


# ---- 문장 쓰기 ---------------------------------------------------------------


async def test_writer_returns_the_llm_sentence_stripped():
    llm = StubLLM("  홍대 근처에서 비 오는 날 저장하신 자물쇠 2건을 찾았습니다.\n")

    text = await HeadlineWriter(llm).write(
        ["홍대 근처", "비 오는 날"], found=2, total=2, suggestions=0
    )

    assert text == "홍대 근처에서 비 오는 날 저장하신 자물쇠 2건을 찾았습니다."
    (call,) = llm.calls
    assert call["system"] == SYSTEM_PROMPT
    assert call["prompt"] == build_prompt(
        ["홍대 근처", "비 오는 날"], found=2, total=2, suggestions=0
    )


async def test_writer_gives_nothing_to_the_llm_but_condition_names_and_counts():
    llm = StubLLM("가을 조건에 맞는 자물쇠 3건을 찾았어요.")

    await HeadlineWriter(llm).write(["가을"], found=3, total=3, suggestions=0)

    # 질문 원문·곡·장소·코멘트를 받을 자리가 아예 없다 — 보내는 글은 위 두 줄이 전부
    assert llm.calls[0]["prompt"] == "질문한 조건: 가을\n조건에 모두 맞는 자물쇠: 3건"


async def test_writer_rejects_a_sentence_with_a_wrong_count():
    llm = StubLLM("가을 조건에 맞는 자물쇠 7건을 찾았어요.")

    assert await HeadlineWriter(llm).write(["가을"], found=3, total=3, suggestions=0) is None


async def test_writer_rejects_an_empty_response():
    assert (
        await HeadlineWriter(StubLLM("  \n")).write(["가을"], found=3, total=3, suggestions=0)
        is None
    )


async def test_llm_errors_are_not_swallowed_here():
    """실패 처리(템플릿으로 대신)는 부르는 쪽 몫이다 — 여기서는 그대로 올려 보낸다."""
    llm = StubLLM(error=RuntimeError("down"))

    with pytest.raises(RuntimeError):
        await HeadlineWriter(llm).write(["가을"], found=3, total=3, suggestions=0)


# ---- 검증 --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "found", "total", "expected"),
    [
        ("가을 조건에 맞는 자물쇠 3건을 찾았어요.", 3, 3, True),
        ("가을 조건에 맞는 자물쇠 세 건을 찾았어요.", 3, 3, False),  # 한글 숫자는 확인할 수 없다
        ("가을 조건에 맞는 자물쇠 13건을 찾았어요.", 3, 3, False),  # 3이 13 안에 들어 있어도 아니다
        ("총 120건 중 상위 50건을 보여드려요.", 50, 120, True),
        ("총 120건이에요.", 50, 120, False),  # 보여주는 건수도 알려야 한다
        ("조건에 모두 맞는 자물쇠는 없어요.", 0, 0, True),
        ("", 3, 3, False),
        ("가" * (MAX_LENGTH + 1) + " 3건", 3, 3, False),
    ],
)
def test_sentence_must_state_the_counts_as_digits(text, found, total, expected):
    assert is_valid(text, found=found, total=total) is expected
