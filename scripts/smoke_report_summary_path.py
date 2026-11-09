"""서비스가 쓰는 길(get_clients → ReportSummarizer)로 가짜 월간 기록 몇 건을 요약해 본다.

로컬 LLM(RunPod의 Gemma)과 Claude 폴백이 실제로 이어지는지 확인하는 용도다. 입력은 이 파일에
적은 가짜 기록뿐이고 실제 사용자 기록은 보내지 않는다. 실제 .env의 키(ANTHROPIC_API_KEY)로 Claude를
부를 수 있다(폴백이 일어날 때만, 건당 Haiku 기준 $0.001 안팎).

어느 쪽이 글을 만들었는지는 호출 횟수로 알려 준다. 환경변수로 설정을 덮어써서 쓴다.
- Gemma 경로 확인(서버가 떠 있어야 함). LOCAL_LLM_URL은 /v1/chat/completions까지 붙인 전체 주소다:
    AI_CLIENT_MODE=real LLM_REPORT_PROVIDER=local \\
    LOCAL_LLM_URL=https://<PodID>-8000.proxy.runpod.net/v1/chat/completions \\
    LOCAL_LLM_MODEL=gemma-3-12b LOCAL_LLM_API_KEY=$VLLM_API_KEY \\
    PYTHONPATH=. python scripts/smoke_report_summary_path.py
- 폴백 경로 확인(서버 주소를 비우거나 틀린 주소로): 위에서 LOCAL_LLM_URL을 빼거나 잘못 적는다.
"""

import argparse
import asyncio
import logging

from app.clients.llm_client import FallbackLLMClient, LLMClient
from app.components.report_summarizer import PhotoAxisStat, ReportSummarizer
from app.config import get_settings
from app.db.repositories.report_repository import ReportStats
from app.dependencies import get_clients

# (기록 줄, 통계, 사진 분위기 비중, 사진 특징) — 전부 가짜 데이터
SCENARIOS = [
    (
        [
            "곡 분위기: 잔잔한 어쿠스틱 발라드 / 아티스트: 아이유 / 날씨: 비 12도 / 장소: 성수동",
            "곡 분위기: 나른한 로파이 팝 / 아티스트: 아이유 / 날씨: 흐림 14도 / 장소: 성수동",
            "곡 분위기: 따뜻한 포크 / 날씨: 비 11도 / 장소: 서울숲 / 사진: 카페",
        ],
        ReportStats(avg_mood_score=None, top_artist="아이유", top_place="성수동"),
        {"카페": 0.6, "자연": 0.4},
        {"분위기": PhotoAxisStat(count=5, ratios={"흐린 회색빛": 0.6, "따뜻한 조명": 0.4})},
    ),
    (
        [
            "곡 분위기: 청량한 신스팝 / 아티스트: 잔나비 / 날씨: 맑음 24도 / 장소: 한강공원",
            "곡 분위기: 경쾌한 인디 록 / 아티스트: 잔나비 / 날씨: 맑음 26도 / 장소: 한강공원",
        ],
        ReportStats(avg_mood_score=None, top_artist="잔나비", top_place="한강공원"),
        {"자연": 0.7, "도시야경": 0.3},
        None,
    ),
    (
        ["곡 분위기: 쓸쓸한 피아노 발라드 / 날씨: 눈 -2도 / 장소: 연남동"],
        ReportStats(avg_mood_score=None, top_artist=None, top_place="연남동"),
        None,
        None,
    ),
]


class CountingLLM:
    """안쪽 LLM을 부른 횟수를 센다. 받은 인자는 그대로 안쪽에 넘긴다."""

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.model = inner.model
        self.calls = 0

    async def complete(self, **kwargs) -> str:
        self.calls += 1
        return await self.inner.complete(**kwargs)


async def main(count: int) -> None:
    settings = get_settings()
    clients = get_clients()
    llm = clients.llm_report
    print(f"모드 {settings.ai_client_mode} | 리포트 제공자 {settings.llm_report_provider}")
    if not isinstance(llm, FallbackLLMClient):
        print("LLM_REPORT_PROVIDER=local이 아니라 폴백 경로를 확인할 수 없어요 (Claude만 부름).")
        return
    primary, fallback = CountingLLM(llm.primary), CountingLLM(llm.fallback)
    llm.primary, llm.fallback = primary, fallback
    summarizer = ReportSummarizer(llm, fallback)
    print(f"기본 {primary.model} / 대체 {fallback.model}\n")

    for index in range(count):
        lines, stats, categories, axes = SCENARIOS[index % len(SCENARIOS)]
        before = (primary.calls, fallback.calls)
        summary = await summarizer.summarize(lines, stats, categories, axes)
        used = (primary.calls - before[0], fallback.calls - before[1])
        print(f"[{index + 1}] 기본 호출 {used[0]}회, 대체 호출 {used[1]}회 | {len(summary)}자")
        print(f"    {summary}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="서비스 경로로 요약 몇 건 해 보기")
    parser.add_argument("--count", type=int, default=3)
    logging.basicConfig(level=logging.WARNING, format="    [로그] %(name)s: %(message)s")
    asyncio.run(main(parser.parse_args().count))
