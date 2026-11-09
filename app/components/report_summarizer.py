"""자물쇠별 요약 줄 + 사진 카테고리 비중·사진 특징 → 1~3문장짜리 AI 리캡 문장.

자물쇠 1건당 한 줄(곡 무드·기분점수·날씨·장소·사진 태그·코멘트를 합친 것)과, 기능2
워커가 저장된 이미지 벡터로 분류한 사진 카테고리 비중, 다축 태그(사진 종류·대상·분위기·
표정·감정)로 센 사진 특징을 모아 그 달에 반복된 패턴을 감성적인 1~3문장(약 140~220자,
최대 300자)으로 뽑아낸다. 화면(리포트 상세)에 카드 문구로 그대로 노출된다. 곡·가수 지식이
필요 없는 언어 작업이라 llm_report(로컬 Gemma 먼저, 실패하면 Claude)를 쓴다.
"""

import logging
import re
from dataclasses import dataclass

from app.clients.llm_client import FakeLLMClient, LLMClient
from app.db.repositories.report_repository import ReportStats
from app.exceptions import LLMError

logger = logging.getLogger("muro.report_summarizer")

PROMPT_VERSION = "report-summary-v9"

# 사진 특징에서 축마다 프롬프트에 싣는 태그 수. 문장이 몇 개 안 되어서 상위 몇 개면 충분하다.
AXIS_TOP_TAGS = 3


@dataclass(frozen=True)
class PhotoAxisStat:
    """다축 태그 한 축의 집계. count는 그 축에서 태그가 정해진 사진 수, ratios는 태그별 비중."""

    count: int
    ratios: dict[str, float]


SYSTEM_PROMPT = """너는 사용자의 한 달 기록을 보고 AI 리캡 한 문장을 만드는 어시스턴트다.
결과는 리포트 화면 카드에 헤드라인 문구로 그대로 노출된다.

[이달의 통계]는 이미 계산된 숫자다 — 그대로 인용해서 언급하고, 다시 계산하거나
추측하지 않는다. 값이 "없음"이면 그 항목은 굳이 언급하지 않는다.
[이달의 기록]에는 자물쇠 1건당 한 줄씩, 곡 분위기·기분점수·날씨·장소·사진·코멘트 중
있는 값만 나열되어 있다. 기분점수는 -50(안 좋음)~50(좋음) 사이 값, 0이 중간값이다.
[이달의 사진 분위기 비중]은 그 달 사진 전체를 분류한 비율이다.
[이달의 사진 특징]은 사진을 사진 종류·대상·분위기(빛·색감)·표정·감정으로 나눈 비율이고,
괄호 안은 그 축에서 분류된 사진 수다. 사진에서 알 수 있는 것은 대상의 종류와 빛·색감,
그림·짤의 표정뿐이다.

- 결과는 "당신은"으로 시작하는 1~3문장이다. 전체는 140~220자 안팎으로 쓰고
  300자를 넘기지 않는다.
- 첫 문장: 이달 가장 뚜렷하게 반복된 날씨·아티스트·장소 중 한두 가지를 장면처럼 쓴다.
  통계·기록에 없는 내용은 지어내지 않는다.
- 둘째 문장: 사진 정보([이달의 사진 분위기 비중], [이달의 사진 특징])가 있으면 빛·색감·
  대상을 풍경을 그리듯 쓴다. 뚜렷하게 앞서는 것만 쓰고, 사진 정보가 없으면 장소나 곡
  분위기로 쓴다.
- 셋째 문장: 한 달을 돌아보는 말로 마무리한다. 새로운 정보는 넣지 않는다. 끝맺음은 아래
  여섯 가지 중 입력에 어울리는 하나를 고르거나, 비슷한 느낌의 다른 표현을 쓴다. 같은
  끝맺음을 매번 쓰지 않는다.
  1) ~하던 날들이었어요  2) ~이 오래 마음에 남을 것 같아요  3) ~ 속에서 이번 달이 천천히
  흘러갔어요  4) ~로 채워진 한 달이었어요  5) ~을 지나 여기까지 왔어요
  6) ~이 조용히 쌓여 갔어요
- 쉼표로 항목을 늘어놓지 않는다. 한 문장에 정보를 세 가지 이상 넣지 않는다.
- 입력에 없는 행동·장소·사물을 덧붙이지 않는다. 입력에 있는 장소 이름·날씨·곡 분위기·
  사진 분류만 근거로 쓴다.
- 감성 표현은 입력에 있는 곡 분위기·날씨·사진의 빛에서만 가져온다. "외로웠던", "행복했던",
  "위로받은"처럼 사용자의 감정을 단정하는 말은 쓰지 않는다.
- 기분점수는 화면에 별도 카드로 이미 표시되니 숫자(예: "32점")나 "기분점수"라는 표현을
  넣지 않는다. 횟수나 비율 같은 숫자도 쓰지 않는다.
- 코멘트가 있으면 참고만 하고 그대로 인용하지 않는다.
- '친구', '가족', '연인'처럼 사진만으로 알 수 없는 관계나 사진 속 사건은 추측하지 않는다.
- 아래 예시는 흐름만 참고한다. (괄호) 자리는 입력의 실제 값으로 바꿔 쓰고, 예시에 쓰인
  낱말과 끝맺음을 그대로 쓰지 않으며, 입력에 없는 값은 쓰지 않는다.
  예시 1: 당신은 (날씨)가 이어지던 이번 달, (아티스트)의 노래를 자주 찾았고 (장소)를 가장
  자주 찾았어요. 사진에는 (사진 장면)이 가장 많이 담겼고 (빛·색감)이 눈에 들어왔어요.
  (곡 분위기)한 노래들이 그 하루하루를 채우던 날들이었어요.
  예시 2: 당신은 (날씨)인 날이면 (곡 분위기)한 노래 쪽으로 마음이 기울었어요. (장소)에서
  보낸 날이 많았고, 사진 속 (빛·색감)이 그달의 공기를 닮아 있었어요. (아티스트)의 목소리가
  오래 마음에 남을 것 같아요.
- 다정한 톤으로 쓴다.
- 위 1~3문장 외의 말은 쓰지 않는다."""

NO_RECORDS_TEXT = "이번 달은 남겨진 기록이 없어요."

# 백엔드 컬럼 ai_recap_text는 정의서(v8.1)에는 VARCHAR(100)이었고, 2026-11-08에 300자로
# 늘리기로 확정됐다. 이걸 넘기면 콜백 전체(배치의 다른 유저 결과까지 포함)가 거부될 수
# 있어서, 프롬프트의 "300자를 넘기지 않는다" 지시를 LLM이 안 지켜도 저장·전송 전에 항상
# 이 길이 이하로 강제한다.
AI_RECAP_TEXT_LIMIT = 300

# v9 실험(Haiku, 가짜 입력 30건)에서 가장 길었던 출력이 211토큰이었다. 여유를 두고 300으로 한다.
MAX_OUTPUT_TOKENS = 300


def _truncate_recap_text(text: str, limit: int = AI_RECAP_TEXT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


RECAP_TEXT_PREFIX = "당신은"
RECAP_MAX_SENTENCES = 3
_BLANK_LINES = re.compile(r"\n\s*\n")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])\s+")
_ENDS_WITH_PUNCTUATION = re.compile(r"[.!?…]$")
_ENGLISH_WORD = re.compile(r"[A-Za-z]{3,}")


def clean_recap_text(raw: str, input_prompt: str) -> str | None:
    """LLM이 쓴 글을 검사해 쓸 수 있는 글로 돌려주고, 쓸 수 없으면 None을 돌려준다.

    - 글 안에 빈 줄로 나뉜 단락이 여러 개면(모델이 혼잣말을 앞에 흘리고 고쳐 쓴 경우) "당신은"으로
      시작하는 마지막 단락만 쓴다. 단락 안의 줄바꿈은 공백으로 바꾼다.
    - "당신은"으로 시작하지 않거나, 입력(프롬프트)에 없는 영어 낱말(3글자 이상)이 있거나, 문장이
      3개를 넘으면 쓸 수 없다. 입력에 있는 영어(가수 이름 등)는 괜찮다.
    - 마지막에 문장부호가 없으면 마침표를 붙인다. Gemma가 "~어요"로 끝내고 마침표를 빼는 일이
      있었다(30건 중 7건). 출력이 길이 한도에서 잘린 경우는 LLM 클라이언트가 이미 오류로 처리해서
      여기까지 오지 않는다.
    곡·장소 이름을 지어내는지, 감정을 단정하는지 같은 내용 검사는 하지 않는다(형식 검사만).
    """
    text = raw.strip()
    if "\n" in text:
        starts = [
            p.strip() for p in _BLANK_LINES.split(text) if p.strip().startswith(RECAP_TEXT_PREFIX)
        ]
        if not starts:
            return None
        text = " ".join(starts[-1].split())
    if not text.startswith(RECAP_TEXT_PREFIX):
        return None
    lowered_input = input_prompt.lower()
    if any(word.lower() not in lowered_input for word in _ENGLISH_WORD.findall(text)):
        return None
    if not _ENDS_WITH_PUNCTUATION.search(text):
        text += "."
    if len(_SENTENCE_BREAK.split(text)) > RECAP_MAX_SENTENCES:
        return None
    return text


def _format_photo_axes(photo_axes: dict[str, PhotoAxisStat]) -> str:
    lines = []
    for axis_name, stat in photo_axes.items():
        ranked = sorted(stat.ratios.items(), key=lambda pair: pair[1], reverse=True)
        tags = ", ".join(f"{tag} {ratio:.0%}" for tag, ratio in ranked[:AXIS_TOP_TAGS])
        lines.append(f"{axis_name}({stat.count}장): {tags}")
    return "\n".join(lines)


class ReportSummarizer:
    """backup_llm: 첫 글이 형식 검사를 통과하지 못했을 때 다시 쓰게 할 LLM(Claude).

    없으면 같은 LLM으로 한 번 더 시도한다. 두 번 다 통과하지 못하면 LLMError를 내서 작업 큐가
    재시도하게 한다. 서버가 꺼져 있는 등 호출 자체가 실패한 경우의 폴백은 llm(FallbackLLMClient)이
    알아서 한다. fake 모드(FakeLLMClient)는 글 형식이 달라 검사하지 않는다.
    """

    def __init__(self, llm: LLMClient, backup_llm: LLMClient | None = None):
        self._llm = llm
        self._backup_llm = backup_llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def summarize(
        self,
        record_lines: list[str],
        stats: ReportStats,
        photo_categories: dict[str, float] | None = None,
        photo_axes: dict[str, PhotoAxisStat] | None = None,
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
        if photo_axes:
            sections.append(f"[이달의 사진 특징]\n{_format_photo_axes(photo_axes)}")

        prompt = "\n\n".join(sections)
        if isinstance(self._llm, FakeLLMClient):
            summary = await self._llm.complete(
                system=SYSTEM_PROMPT, prompt=prompt, max_tokens=MAX_OUTPUT_TOKENS
            )
            return _truncate_recap_text(summary)

        for attempt, client in enumerate((self._llm, self._backup_llm or self._llm), start=1):
            raw = await client.complete(
                system=SYSTEM_PROMPT, prompt=prompt, max_tokens=MAX_OUTPUT_TOKENS
            )
            cleaned = clean_recap_text(raw, prompt)
            if cleaned is not None:
                return _truncate_recap_text(cleaned)
            logger.warning(
                "요약 글이 형식 검사를 통과하지 못했어요 (모델 %s, %d번째 시도)",
                client.model,
                attempt,
            )
        raise LLMError("AI 리캡 문장이 형식 검사를 두 번 통과하지 못했어요.")
