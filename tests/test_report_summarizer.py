import pytest

from app.clients.llm_client import FakeLLMClient
from app.components.report_summarizer import (
    AI_RECAP_TEXT_LIMIT,
    AXIS_TOP_TAGS,
    MAX_OUTPUT_TOKENS,
    NO_RECORDS_TEXT,
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    PhotoAxisStat,
    ReportSummarizer,
    clean_recap_text,
)
from app.db.repositories.report_repository import ReportStats
from app.exceptions import LLMError

STATS = ReportStats(avg_mood_score=None, top_artist=None, top_place=None)


class ScriptedLLM:
    model = "scripted-llm"

    def __init__(self, response: str):
        self._response = response

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        return self._response


async def test_summary_over_backend_limit_is_truncated_with_ellipsis():
    """백엔드 ai_recap_text 컬럼은 300자 — 넘기면 콜백 전체가 거부될 수 있다.

    LLM이 "300자를 넘기지 않는다" 프롬프트 지시를 안 지키고 길게 쓴 경우를 시뮬레이션한다.
    """
    assert AI_RECAP_TEXT_LIMIT == 300
    long_response = "당신은 " + "가" * (AI_RECAP_TEXT_LIMIT + 50) + "."
    summarizer = ReportSummarizer(ScriptedLLM(long_response))

    summary = await summarizer.summarize(["곡 분위기: 잔잔한"], STATS)

    assert len(summary) == AI_RECAP_TEXT_LIMIT
    assert summary.endswith("…")


async def test_summary_within_limit_is_untouched():
    summarizer = ReportSummarizer(ScriptedLLM("당신은 잔잔한 곡을 즐겨 들었어요."))

    summary = await summarizer.summarize(["곡 분위기: 잔잔한"], STATS)

    assert summary == "당신은 잔잔한 곡을 즐겨 들었어요."


async def test_no_records_and_no_photos_skips_llm_call_entirely():
    summarizer = ReportSummarizer(ScriptedLLM("가" * 150))

    summary = await summarizer.summarize([], STATS)

    assert summary == NO_RECORDS_TEXT


class RecordingLLM(ScriptedLLM):
    def __init__(self):
        super().__init__("당신은 잔잔한 곡을 즐겨 들었어요.")
        self.system = ""
        self.prompt = ""
        self.max_tokens = 0

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.system = system
        self.prompt = prompt
        self.max_tokens = max_tokens
        return await super().complete(system=system, prompt=prompt, max_tokens=max_tokens)


async def test_v9_prompt_and_output_budget_are_sent_to_the_llm():
    llm = RecordingLLM()

    await ReportSummarizer(llm).summarize(["곡 분위기: 잔잔한"], STATS)

    assert PROMPT_VERSION == "report-summary-v9"
    assert llm.system == SYSTEM_PROMPT
    assert llm.max_tokens == MAX_OUTPUT_TOKENS == 300
    flat = " ".join(SYSTEM_PROMPT.split())  # 프롬프트는 줄바꿈으로 나뉘어 있어서 한 줄로 편다
    assert '"당신은"으로 시작하는 1~3문장' in flat
    assert "300자를 넘기지 않는다" in flat
    assert "한 문장만" not in flat
    # 실험에서 모델이 혼잣말을 흘리게 만든 금지 예시 단어는 프롬프트에 넣지 않는다.
    assert "골목" not in flat


async def test_photo_axes_go_into_the_prompt_as_their_own_section():
    llm = RecordingLLM()
    photo_axes = {
        "사진 종류": PhotoAxisStat(count=5, ratios={"진짜 사진": 0.8, "그림·짤": 0.2}),
        "대상": PhotoAxisStat(count=4, ratios={"여러 사람": 0.75, "음식·음료": 0.25}),
    }

    await ReportSummarizer(llm).summarize(["곡 분위기: 잔잔한"], STATS, {"카페": 1.0}, photo_axes)

    assert "[이달의 사진 특징]" in llm.prompt
    assert "사진 종류(5장): 진짜 사진 80%, 그림·짤 20%" in llm.prompt
    assert "대상(4장): 여러 사람 75%, 음식·음료 25%" in llm.prompt


async def test_prompt_has_no_photo_feature_section_without_axes():
    llm = RecordingLLM()

    await ReportSummarizer(llm).summarize(["곡 분위기: 잔잔한"], STATS, {"카페": 1.0})

    assert "[이달의 사진 특징]" not in llm.prompt
    assert "[이달의 사진 분위기 비중]" in llm.prompt  # 기존 장소 비중은 그대로


async def test_photo_axes_list_only_the_top_tags_in_order():
    llm = RecordingLLM()
    ratios = {"가": 0.1, "나": 0.4, "다": 0.3, "라": 0.2}
    assert len(ratios) > AXIS_TOP_TAGS

    await ReportSummarizer(llm).summarize(
        ["곡 분위기: 잔잔한"], STATS, None, {"분위기": PhotoAxisStat(count=10, ratios=ratios)}
    )

    assert "분위기(10장): 나 40%, 다 30%, 라 20%" in llm.prompt
    assert "가 10%" not in llm.prompt


# ---- 형식 검사(clean_recap_text) ----

INPUT = "[이달의 통계]\n가장 많이 들은 아티스트 Daniel Caesar, 가장 많이 간 장소 성수동\n"
GOOD = "당신은 비 오는 날이면 잔잔한 노래에 마음을 기울였어요. 성수동에서 보낸 날이 많았어요."


def test_good_text_passes_unchanged():
    assert clean_recap_text(GOOD, INPUT) == GOOD


def test_missing_final_period_is_added():
    assert clean_recap_text(GOOD.rstrip("."), INPUT) == GOOD
    assert (
        clean_recap_text(GOOD[:-1] + "!", INPUT) == GOOD[:-1] + "!"
    )  # 이미 문장부호가 있으면 그대로


def test_english_word_from_the_input_is_allowed_but_stray_english_is_not():
    with_artist = "당신은 Daniel Caesar의 노래를 자주 찾았어요."
    assert clean_recap_text(with_artist, INPUT) == with_artist
    assert (
        clean_recap_text("당신은 daniel caesar의 노래를 자주 찾았어요.", INPUT) is not None
    )  # 대소문자 무시
    assert clean_recap_text("당신은 I must avoid inventing details.", INPUT) is None
    assert clean_recap_text(GOOD + " Wait", INPUT) is None


@pytest.mark.parametrize(
    "text",
    [
        "",
        "안녕하세요. 이번 달은 비가 많았어요.",  # "당신은"으로 시작하지 않음
        GOOD + " 사진에는 노을이 담겼어요. 오래 마음에 남을 것 같아요.",  # 4문장
    ],
)
def test_text_outside_the_format_is_rejected(text):
    assert clean_recap_text(text, INPUT) is None


def test_three_sentences_are_fine():
    text = "당신은 비를 좋아했어요. 사진에는 노을이 담겼어요. 오래 마음에 남을 것 같아요."
    assert clean_recap_text(text, INPUT) == text


def test_reasoning_before_the_answer_is_dropped_keeping_the_last_recap_paragraph():
    """실험(Haiku)에서 모델이 영어 혼잣말을 흘리고 고쳐 쓴 글이 실제로 있었다."""
    leaked = (
        "당신은 자우림의 노래와 한남동의 골목 같은 하루를... \n\n"
        'Wait, I must avoid inventing details like "골목" (not in input).\n\n'
        "당신은 자우림의 노래가 자주 흐르던 이번 달, 한남동을 가장 자주 찾았어요."
    )

    assert clean_recap_text(leaked, INPUT) == (
        "당신은 자우림의 노래가 자주 흐르던 이번 달, 한남동을 가장 자주 찾았어요."
    )


def test_single_line_breaks_inside_the_answer_become_spaces():
    text = "당신은 비를 좋아했어요.\n사진에는 노을이 담겼어요."
    assert clean_recap_text(text, INPUT) == "당신은 비를 좋아했어요. 사진에는 노을이 담겼어요."


def test_paragraphs_without_a_recap_start_are_rejected():
    assert clean_recap_text("생각해 볼게요.\n\n다시 쓸게요.", INPUT) is None


# ---- 요약기 흐름: 검사 실패 시 Claude로 다시 쓰기 ----


class SequenceLLM:
    def __init__(self, model, *answers):
        self.model = model
        self._answers = list(answers)
        self.calls = 0

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.calls += 1
        answer = self._answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


async def test_period_is_added_to_the_saved_summary():
    summarizer = ReportSummarizer(SequenceLLM("gemma", "당신은 잔잔한 곡을 즐겨 들었어요"))

    assert await summarizer.summarize(["곡 분위기: 잔잔한"], STATS) == (
        "당신은 잔잔한 곡을 즐겨 들었어요."
    )


async def test_text_failing_the_check_is_rewritten_by_the_backup_llm():
    gemma = SequenceLLM("gemma", "한 달 동안 비가 왔어요.")
    claude = SequenceLLM("haiku", "당신은 비 오는 날 잔잔한 노래를 찾았어요.")
    summarizer = ReportSummarizer(gemma, claude)

    summary = await summarizer.summarize(["곡 분위기: 잔잔한"], STATS)

    assert summary == "당신은 비 오는 날 잔잔한 노래를 찾았어요."
    assert (gemma.calls, claude.calls) == (1, 1)


async def test_good_text_never_calls_the_backup_llm():
    gemma = SequenceLLM("gemma", GOOD)
    claude = SequenceLLM("haiku")
    summarizer = ReportSummarizer(gemma, claude)

    assert await summarizer.summarize(["곡 분위기: 잔잔한"], STATS) == GOOD
    assert claude.calls == 0


async def test_without_a_backup_the_same_llm_gets_one_more_try():
    llm = SequenceLLM("claude", "형식이 틀린 글", GOOD)

    assert await ReportSummarizer(llm).summarize(["곡 분위기: 잔잔한"], STATS) == GOOD
    assert llm.calls == 2


async def test_two_failed_checks_raise_llm_error_so_the_job_queue_retries():
    gemma = SequenceLLM("gemma", "틀린 글 1")
    claude = SequenceLLM("haiku", "틀린 글 2")

    with pytest.raises(LLMError, match="형식 검사"):
        await ReportSummarizer(gemma, claude).summarize(["곡 분위기: 잔잔한"], STATS)
    assert (gemma.calls, claude.calls) == (1, 1)


async def test_llm_error_from_the_client_is_not_swallowed():
    llm = SequenceLLM("claude", LLMError("Claude 장애"))

    with pytest.raises(LLMError, match="Claude 장애"):
        await ReportSummarizer(llm, SequenceLLM("haiku", GOOD)).summarize(
            ["곡 분위기: 잔잔한"], STATS
        )


async def test_fake_mode_skips_the_check_and_only_truncates():
    summary = await ReportSummarizer(FakeLLMClient()).summarize(["곡 분위기: 잔잔한"], STATS)

    assert summary.startswith("설명: [fake]")  # "당신은"으로 시작하지 않아도 그대로 쓴다
    assert len(summary) <= AI_RECAP_TEXT_LIMIT
