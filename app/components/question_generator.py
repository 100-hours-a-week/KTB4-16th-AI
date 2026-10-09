"""오늘의 질문 생성 — 장르 그룹별 A vs B 취향 질문을 LLM으로 만든다.

질문 하나에 A·B 답이 반반 갈려야 서로 다른 답을 고른 사람끼리 매칭이 된다. 그래서
사실 퀴즈("아이유 대표곡은?")가 아니라 정답이 없는 취향 질문만 만든다.

LLM이 아무 곡이나 지어내지 않도록 우리 사용자들이 실제로 많이 저장한 가수·무드·장소·
날씨 통계를 재료로 준다. 통계가 비어 있으면(초반, 데이터 적음) 장르 일반 지식으로 만든다.
"""

import re

from app.clients.llm_client import LLMClient
from app.components.music_tags import COMMON_GROUP
from app.db.repositories.balance_stats_repository import GenreStats

PROMPT_VERSION = "balance-question-v1"
QUESTION_MAX_LEN = 50
OPTION_MAX_LEN = 20

SYSTEM_PROMPT = f"""너는 음악 취향 투표 앱 MULO의 '오늘의 질문'을 만든다.
사용자는 질문에 A 또는 B로 투표하고, 다른 답을 고른 사람과 1:1로 채팅한다.

규칙:
- 정답이 없는 취향 질문만 만든다. 사실 퀴즈(누구의 대표곡은?, 몇 년 발매?)는 만들지 않는다.
- A와 B가 반반 갈릴 만큼 비슷하게 매력적이어야 한다. 한쪽이 뻔히 이기는 질문은 만들지 않는다.
- 음악을 듣는 상황·감정·취향에 대한 질문으로 만든다. 채팅을 시작하기 좋은 가벼운 주제로.
- 특정 집단 비하, 정치, 종교, 성적인 주제, 외모 평가는 다루지 않는다.
- 가수 이름은 '재료'에 나온 가수만 쓴다. 재료에 없는 가수나 곡 제목은 지어내지 않는다.
- 질문은 {QUESTION_MAX_LEN}자 이내, 선택지는 각각 {OPTION_MAX_LEN}자 이내로 짧게.
- 서로 겹치지 않는 질문을 만든다.

출력 형식(한 줄에 질문 하나, 다른 말은 붙이지 않는다):
<질문> | <선택지 A> | <선택지 B>"""


class QuestionGenerator:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def generate(
        self, music_genre: str, stats: GenreStats, count: int
    ) -> list[tuple[str, str, str]]:
        # 형식이 깨진 줄은 버리므로 조금 넉넉히 요청한다
        prompt = _build_prompt(music_genre, stats, count + 2)
        response = await self._llm.complete(system=SYSTEM_PROMPT, prompt=prompt, max_tokens=1024)
        return parse(response)[:count]


def _build_prompt(music_genre: str, stats: GenreStats, count: int) -> str:
    target = (
        "모든 사용자 공통 (장르 구분 없음)"
        if music_genre == COMMON_GROUP
        else f"'{music_genre}' 장르를 좋아하는 사용자"
    )
    lines = [f"대상: {target}", f"만들 개수: {count}개", ""]
    if stats.record_count == 0:
        lines.append("재료: 아직 저장된 기록이 없다. 가수 이름 없이 일반적인 취향 질문으로 만든다.")
    else:
        lines.append(f"재료 (최근 사용자들이 저장한 음악 기록 {stats.record_count}건 집계):")
        lines += [
            _stat_line("많이 저장된 가수", stats.top_artists),
            _stat_line("많이 붙은 무드", stats.top_moods),
            _stat_line("많이 저장된 장소", stats.top_places),
            _stat_line("많이 저장된 날씨", stats.top_weathers),
        ]
    return "\n".join(line for line in lines if line is not None)


def _stat_line(label: str, items: list[tuple[str, int]]) -> str | None:
    if not items:
        return None
    return f"- {label}: " + ", ".join(f"{name}({cnt})" for name, cnt in items)


_NUMBERING = re.compile(r"^\s*(\d+[.)]|[-*•])\s*")


def parse(text: str) -> list[tuple[str, str, str]]:
    """'질문 | A | B' 줄만 살린다. 길이 초과·A와 B가 같은 줄·중복 질문은 버린다."""
    seen: set[str] = set()
    out: list[tuple[str, str, str]] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in _NUMBERING.sub("", line).split("|")]
        if len(parts) != 3 or not all(parts):
            continue
        question, a, b = parts
        if "<" in question or len(question) > QUESTION_MAX_LEN:
            continue
        if len(a) > OPTION_MAX_LEN or len(b) > OPTION_MAX_LEN or a == b:
            continue
        if question in seen:
            continue
        seen.add(question)
        out.append((question, a, b))
    return out
