from app.components.report_summarizer import (
    AI_RECAP_TEXT_LIMIT,
    NO_RECORDS_TEXT,
    ReportSummarizer,
)
from app.db.repositories.report_repository import ReportStats

STATS = ReportStats(avg_mood_score=None, top_artist=None, top_place=None)


class ScriptedLLM:
    model = "scripted-llm"

    def __init__(self, response: str):
        self._response = response

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        return self._response


async def test_summary_over_backend_limit_is_truncated_with_ellipsis():
    """백엔드 ai_recap_text 컬럼이 VARCHAR(100) — 넘기면 콜백 전체가 거부된다.

    LLM이 "한 문장만" 프롬프트 지시를 안 지키고 길게 쓴 경우를 시뮬레이션한다.
    """
    long_response = "가" * 150
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
