import pytest

from app.components.answer_headline import (
    MAX_LENGTH,
    SYSTEM_PROMPT,
    HeadlineWriter,
    build_prompt,
    is_valid,
)
from tests.memory_fakes import StubLLM

# ---- LLM에 보내는 글: 조건 이름과 결과 유무만, 건수 숫자는 보내지 않는다 ------------------


def test_prompt_has_only_condition_names_and_whether_there_are_results():
    prompt = build_prompt(["홍대 근처", "비 오는 날"], found=2, total=2, suggestions=0)

    assert prompt == "질문한 조건: 홍대 근처, 비 오는 날\n조건에 모두 맞는 자물쇠: 있음"


def test_prompt_says_there_are_many_when_only_some_are_shown():
    prompt = build_prompt(["가을"], found=50, total=120, suggestions=0)

    assert prompt.splitlines()[1] == "조건에 모두 맞는 자물쇠: 많아서 일부만 보여줌"


def test_prompt_tells_when_nothing_matches_all_conditions_but_some_match_in_part():
    prompt = build_prompt(["가을", "서울숲 근처"], found=0, total=0, suggestions=3)

    assert prompt.splitlines()[1:] == [
        "조건에 모두 맞는 자물쇠: 없음",
        "일부만 맞는 자물쇠: 있음 (화면에 따로 보여줌)",
    ]


@pytest.mark.parametrize(
    ("found", "total", "suggestions"),
    [(2, 2, 0), (50, 120, 0), (0, 0, 3), (4, 4, 3)],
)
def test_prompt_never_contains_a_count(found, total, suggestions):
    """건수가 LLM 글에 섞여 들어가면 문장에 숫자가 나올 수 있다 — 아예 보내지 않는다."""
    prompt = build_prompt(["가을"], found=found, total=total, suggestions=suggestions)

    assert not any(ch.isdigit() for ch in prompt)


def test_system_prompt_forbids_inventing_facts_listing_records_and_counts():
    assert "지어내지 않는다" in SYSTEM_PROMPT
    assert "나열하지 않는다" in SYSTEM_PROMPT
    assert "건수는 말하지 않는다" in SYSTEM_PROMPT


# ---- 문장 쓰기 ---------------------------------------------------------------


async def test_writer_returns_the_llm_sentence_stripped():
    llm = StubLLM("  홍대 근처에서 비 오는 날 저장하신 자물쇠를 찾았습니다.\n")

    text = await HeadlineWriter(llm).write(
        ["홍대 근처", "비 오는 날"], found=2, total=2, suggestions=0
    )

    assert text == "홍대 근처에서 비 오는 날 저장하신 자물쇠를 찾았습니다."
    (call,) = llm.calls
    assert call["system"] == SYSTEM_PROMPT
    assert call["prompt"] == build_prompt(
        ["홍대 근처", "비 오는 날"], found=2, total=2, suggestions=0
    )


async def test_writer_gives_nothing_to_the_llm_but_condition_names_and_existence():
    llm = StubLLM("가을 조건에 맞는 자물쇠를 찾았어요.")

    await HeadlineWriter(llm).write(["가을"], found=3, total=3, suggestions=0)

    # 질문 원문·곡·장소·코멘트·건수를 받을 자리가 아예 없다 — 보내는 글은 위 두 줄이 전부
    assert llm.calls[0]["prompt"] == "질문한 조건: 가을\n조건에 모두 맞는 자물쇠: 있음"


async def test_writer_rejects_a_sentence_that_states_a_count():
    llm = StubLLM("가을 조건에 맞는 자물쇠 3건을 찾았어요.")

    assert await HeadlineWriter(llm).write(["가을"], found=3, total=3, suggestions=0) is None


async def test_writer_accepts_digits_that_belong_to_a_condition_name():
    llm = StubLLM("3월 조건에 맞는 자물쇠를 찾았어요.")

    text = await HeadlineWriter(llm).write(["3월"], found=3, total=3, suggestions=0)

    assert text == "3월 조건에 맞는 자물쇠를 찾았어요."


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


# ---- 검증: 건수 숫자가 섞이면 거절, 조건 이름의 숫자는 허용 ---------------------------


@pytest.mark.parametrize(
    ("text", "labels", "expected"),
    [
        ("가을 조건에 맞는 자물쇠를 찾았어요.", ["가을"], True),
        ("조건에 모두 맞는 자물쇠는 없어요.", ["가을"], True),
        ("가을 조건에 맞는 자물쇠가 많아서 일부만 보여드려요.", ["가을"], True),
        ("가을 조건에 맞는 자물쇠 3건을 찾았어요.", ["가을"], False),  # 건수 숫자
        ("가을 조건에 맞는 자물쇠 13건이에요.", ["가을"], False),
        ("총 120건 중 상위 50건을 보여드려요.", ["가을"], False),
        ("3월 조건에 맞는 자물쇠를 찾았어요.", ["3월"], True),  # 조건 이름에 원래 있는 숫자
        (
            "3월 조건에 맞는 자물쇠 3건을 찾았어요.",
            ["3월"],
            False,
        ),  # 이름의 숫자(3월)만 허용, 3건은 아님
        ("2025년 가을, 3월에 맞는 자물쇠를 찾았어요.", ["2025년 가을", "3월"], True),
        ("가을 조건에 맞는 자물쇠 한 건을 찾았어요.", ["가을"], False),  # 한글 개수 표현
        ("가을 조건에 맞는 자물쇠 두 건을 찾았어요.", ["가을"], False),
        ("가을 조건에 맞는 자물쇠가 몇 개 있어요.", ["가을"], False),
        (
            "가을 조건에 모두 맞는 자물쇠를 찾았어요.",
            ["가을"],
            True,
        ),  # "조건"의 "건"은 개수가 아니다
        ("", ["가을"], False),
        ("가" * (MAX_LENGTH + 1), ["가을"], False),
    ],
)
def test_sentence_must_not_state_a_count(text, labels, expected):
    assert is_valid(text, labels) is expected
