"""자물쇠별 요약 줄 + 사진 카테고리 비중 → 한 줄짜리 AI 리캡 문장.

자물쇠 1건당 한 줄(곡 무드·기분점수·날씨·장소·사진 태그·코멘트를 합친 것)과, 기능2
워커가 저장된 이미지 벡터로 분류한 사진 카테고리 비중을 모아 그 달에 반복된 패턴을
한 문장으로 뽑아낸다. 화면(리포트 상세)에 카드 헤드라인으로 그대로 노출되는 문구라
길게 쓰지 않는다. 곡·가수 지식이 필요 없는 언어 작업이라 llm_general(곧 로컬로
옮길 갈래)을 쓴다.
"""

from app.clients.llm_client import LLMClient
from app.db.repositories.report_repository import ReportStats

PROMPT_VERSION = "report-summary-v6"


SYSTEM_PROMPT = """너는 사용자의 한 달 기록을 보고 AI 리캡 한 문장을 만드는 어시스턴트다.
결과는 리포트 화면 카드에 헤드라인 문구로 그대로 노출된다.

[이달의 통계]는 이미 계산된 숫자다 — 그대로 인용해서 언급하고, 다시 계산하거나
추측하지 않는다. 값이 "없음"이면 그 항목은 굳이 언급하지 않는다.
[이달의 기록]에는 자물쇠 1건당 한 줄씩, 곡 분위기·기분점수·날씨·장소·사진·코멘트 중
있는 값만 나열되어 있다. 기분점수는 -50(안 좋음)~50(좋음) 사이 값, 0이 중간값이다.
[이달의 사진 분위기 비중]은 그 달 사진 전체를 분류한 비율이다.

- 결과는 "당신은"으로 시작하는 **한 문장**만 쓴다. 두 문장 이상 쓰지 않는다.
- 이 사람이 이번 달 반복적으로 보인 패턴(자주 겹친 날씨·장소·곡 분위기의 조합)을
  짚어서 요약한다. 통계·개별 기록에 없는 내용은 지어내지 않는다.
- 기분점수는 화면에 별도 카드로 이미 표시되니, 문장에 숫자(예: "32점")나
  "기분점수"라는 표현을 넣지 않는다. 대신 분위기를 말로 풀어서 표현한다.
- 좋은 예: "당신은 주로 비 오는 날 잔잔한 곡을, 흐린 저녁의 동탄역에서 저장했어요."
- 나쁜 예: 3문장 이상으로 늘어놓거나, 기분점수 숫자를 언급하거나, 데이터에 없는
  감정을 추측해서 덧붙이는 것.
- 뚜렷하게 반복되는 패턴이 안 보이면, 있는 정보 중 가장 눈에 띄는 것 위주로 자연스럽게
  한 문장으로 압축한다.
- 코멘트가 있으면 참고만 하고 그대로 인용하지 않는다.
- 다정하고 담백한 톤으로 쓴다.
- 한 문장 외의 말은 쓰지 않는다."""

NO_RECORDS_TEXT = "이번 달은 남겨진 기록이 없어요."


class ReportSummarizer:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def summarize(
        self,
        record_lines: list[str],
        stats: ReportStats,
        photo_categories: dict[str, float] | None = None,
    ) -> str:
        """자물쇠 요약 줄과 사진 카테고리가 둘 다 없으면 LLM을 부르지 않는다."""
        photo_categories = photo_categories or {}
        if not record_lines and not photo_categories:
            return NO_RECORDS_TEXT

        sections = []
        stat_parts = []
        if stats.top_artist:
            stat_parts.append(f"가장 많이 들은 아티스트 {stats.top_artist}")
        if stats.top_place:
            stat_parts.append(f"가장 많이 간 장소 {stats.top_place}")
        if stat_parts:
            sections.append(f"[이달의 통계]\n{', '.join(stat_parts)}")
        if record_lines:
            lines = "\n".join(f"{i}. {line}" for i, line in enumerate(record_lines, start=1))
            sections.append(f"[이달의 기록]\n{lines}")
        if photo_categories:
            ranked = sorted(photo_categories.items(), key=lambda pair: pair[1], reverse=True)
            category_line = ", ".join(f"{name} {ratio:.0%}" for name, ratio in ranked)
            sections.append(f"[이달의 사진 분위기 비중]\n{category_line}")

        prompt = "\n\n".join(sections)
        return await self._llm.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=150)
