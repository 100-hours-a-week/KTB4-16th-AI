from app.components.report_summarizer import (
    AI_RECAP_TEXT_LIMIT,
    AXIS_TOP_TAGS,
    NO_RECORDS_TEXT,
    PhotoAxisStat,
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


class RecordingLLM(ScriptedLLM):
    def __init__(self):
        super().__init__("당신은 잔잔한 곡을 즐겨 들었어요.")
        self.prompt = ""

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.prompt = prompt
        return await super().complete(system=system, prompt=prompt, max_tokens=max_tokens)


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
