"""기억 검색 답변 앞머리 문장 — LLM이 조건 이름과 건수만 보고 쓴다.

자물쇠 목록은 프론트가 matchedRecords로 따로 보여주므로 문장은 "어떤 조건으로 몇 건을 찾았는지"만
말한다. 형식과 말투는 LLM이 자유롭게 정한다.

LLM에는 질문에서 규칙이 읽어 낸 조건 이름(예: "홍대 근처", "비 오는 날")과 숫자만 보낸다. 질문
원문, 곡 제목, 가수, 장소, 날짜, 코멘트는 보내지 않는다. 그래서 LLM이 틀릴 수 있는 것은 조건 이름과
숫자를 말로 옮기는 부분뿐이고, 숫자만은 문장에 그대로 들어 있는지 확인한다. 어긋나면 None을
돌려주고, 부른 쪽이 템플릿 문장(memory_answer.template_headline)으로 대신한다.
"""

import re
from collections.abc import Sequence

from app.clients.llm_client import LLMClient

PROMPT_VERSION = "memory-headline-v1"

MAX_LENGTH = 200  # 앞머리는 1~2문장이다. 이보다 길면 목록을 풀어 쓰는 등 지시를 어긴 것으로 본다

SYSTEM_PROMPT = """너는 음악 기록 앱 '뮤로'의 검색 결과 안내 문장을 쓴다.
사용자가 저장해 둔 기록을 '자물쇠'라고 부른다.

규칙:
- 1~2문장으로, 정중한 존댓말로 쓴다.
- 질문한 조건 이름들을 자연스러운 말로 이어서 말하고, 조건에 모두 맞는 자물쇠가 몇 건인지 알린다.
- 건수는 반드시 아라비아 숫자로 쓴다(예: 2건). 받은 숫자를 바꾸거나 새 숫자를 만들지 않는다.
- 받은 정보에 없는 내용(곡, 가수, 장소, 날짜, 분위기)을 지어내지 않는다.
- 자물쇠 목록은 따로 보여주므로 하나하나 나열하지 않는다.
- 조건에 모두 맞는 자물쇠가 없으면 없다고 알린다.
- 일부만 맞는 자물쇠가 있다고 알려 주면 마지막에 비슷한 자물쇠도 함께 보여준다고 짧게 덧붙인다.
- 이모지와 따옴표 없이 문장만 출력한다."""


class HeadlineWriter:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def write(
        self, labels: Sequence[str], *, found: int, total: int, suggestions: int
    ) -> str | None:
        """앞머리 문장. 쓸 수 없는 응답이면 None.

        labels: 조건 이름들. found: 결과로 보내는 건수. total: 조건에 모두 맞는 전체 건수.
        suggestions: 문장 아래에 제안하는 일부만 맞는 건수.
        """
        response = await self._llm.complete(
            system=SYSTEM_PROMPT,
            prompt=build_prompt(labels, found=found, total=total, suggestions=suggestions),
            max_tokens=200,
        )
        text = response.strip()
        return text if is_valid(text, found=found, total=total) else None


def build_prompt(labels: Sequence[str], *, found: int, total: int, suggestions: int) -> str:
    lines = [f"질문한 조건: {', '.join(labels)}"]
    if found == 0:
        lines.append("조건에 모두 맞는 자물쇠: 0건 (없음)")
    elif total > found:
        lines.append(f"조건에 모두 맞는 자물쇠: 총 {total}건 (그중 {found}건을 보여줌)")
    else:
        lines.append(f"조건에 모두 맞는 자물쇠: {found}건")
    if suggestions:
        lines.append(f"일부만 맞는 자물쇠: {suggestions}건 (화면에 따로 보여줌)")
    return "\n".join(lines)


def is_valid(text: str, *, found: int, total: int) -> bool:
    """비어 있지 않고, 너무 길지 않고, 알려야 할 건수가 숫자 그대로 들어 있어야 한다."""
    if not text or len(text) > MAX_LENGTH:
        return False
    if found == 0:
        return True  # 건수를 말할 것이 없다 ("없어요")
    wanted = {str(found)} if total <= found else {str(found), str(total)}
    return wanted <= set(re.findall(r"\d+", text))
