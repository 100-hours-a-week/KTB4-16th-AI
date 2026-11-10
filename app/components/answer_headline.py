"""기억 검색 답변 앞머리 문장 — LLM이 조건 이름과 결과 유무만 보고 쓴다.

자물쇠 목록과 건수는 프론트가 records로 따로 보여주므로 문장은 "어떤 조건으로 찾았는지, 찾았는지"만
말한다. **건수 숫자는 쓰지 않는다.** 말투는 따뜻하고 감성적인 존댓말이다(memory-headline-v3).
LLM이 받는 정보는 그대로라서 감성은 말투에만 있고, 조건에 없는 사실을 더하지 않는다.

LLM에는 질문에서 규칙이 읽어 낸 조건 이름(예: "홍대 근처", "비 오는 날")과 결과 유무(있음·없음·
많아서 일부만)만 보낸다. 질문 원문, 곡 제목, 가수, 장소, 날짜, 코멘트, 건수는 보내지 않는다.
그래서 LLM이 틀릴 수 있는 것은 조건 이름을 말로 옮기는 부분뿐이다. 건수 숫자가 섞이면
(조건 이름에 원래 있는 숫자("3월")는 허용) None을 돌려주고, 부른 쪽이 템플릿 문장
(memory_answer.template_headline)으로 대신한다.
"""

import re
from collections.abc import Sequence

from langsmith import traceable

from app.clients.llm_client import LLMClient

PROMPT_VERSION = "memory-headline-v3"

MAX_LENGTH = 200  # 앞머리는 1~2문장이다. 이보다 길면 목록을 풀어 쓰는 등 지시를 어긴 것으로 본다

SYSTEM_PROMPT = """너는 음악 기록 앱 '뮤로'의 검색 결과 안내 문장을 쓴다.
사용자가 저장해 둔 기록을 '자물쇠'라고 부른다. 이 문장은 사용자가 남겨 둔 기억을 다시 만나게 해 주는
첫마디다.

말투:
- 따뜻하고 감성적인 존댓말 1~2문장으로 쓴다. 그날을 떠올리게 하는 부드러운 말을 한마디 곁들인다.
- 과장하거나 시처럼 길게 쓰지 않는다. 담백한 감성이면 충분하다.
  예) 조건이 "겨울, 한강 근처"이고 찾았다면:
  겨울 한강 근처에서 남겨 둔 기억을 찾았어요. 그날의 기억이 노래와 함께 다시 떠오르길 바라요.
  예시는 말투만 참고하고 문장을 그대로 베끼지 않는다.

규칙:
- 질문한 조건 이름들을 자연스러운 말로 이어서 말하고, 조건에 모두 맞는 자물쇠를 찾았는지 알린다.
- **건수는 말하지 않는다.** 몇 건, 몇 개, 한 건, 두 건 같은 숫자 표현을 쓰지 않는다.
  받은 조건 이름에 숫자가 들어 있으면(예: 3월) 그 말만 그대로 쓴다.
- 받은 정보에 없는 내용(곡, 가수, 장소, 날짜, 날씨, 분위기, 사람)을 지어내지 않는다.
  감정을 담는 것은 괜찮지만 사실을 더하지는 않는다.
- 자물쇠 목록은 따로 보여주므로 하나하나 나열하지 않는다.
- 조건에 모두 맞는 자물쇠가 없으면 없다고 다정하게 알린다.
- "많아서 일부만 보여줌"이라고 받으면 조건에 맞는 자물쇠가 많아서 일부만 보여드린다고 알린다.
- 일부만 맞는 자물쇠가 있다고 알려 주면 마지막에 비슷한 자물쇠도 함께 보여준다고 짧게 덧붙인다.
- 이모지와 따옴표 없이 문장만 출력한다."""


class HeadlineWriter:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    @traceable(name="headline_llm", run_type="llm")
    async def write(
        self, labels: Sequence[str], *, found: int, total: int, suggestions: int
    ) -> str | None:
        """앞머리 문장. 쓸 수 없는 응답이면 None.

        labels: 조건 이름들. found·total·suggestions는 결과가 있는지, 많아서 일부만 보내는지,
        제안이 있는지를 정하는 데만 쓰고 숫자는 LLM에 보내지 않는다.
        """
        response = await self._llm.complete(
            system=SYSTEM_PROMPT,
            prompt=build_prompt(labels, found=found, total=total, suggestions=suggestions),
            max_tokens=200,
        )
        text = response.strip()
        return text if is_valid(text, labels) else None


def build_prompt(labels: Sequence[str], *, found: int, total: int, suggestions: int) -> str:
    lines = [f"질문한 조건: {', '.join(labels)}"]
    if found == 0:
        lines.append("조건에 모두 맞는 자물쇠: 없음")
    elif total > found:
        lines.append("조건에 모두 맞는 자물쇠: 많아서 일부만 보여줌")
    else:
        lines.append("조건에 모두 맞는 자물쇠: 있음")
    if suggestions:
        lines.append("일부만 맞는 자물쇠: 있음 (화면에 따로 보여줌)")
    return "\n".join(lines)


# 한국어로 쓴 개수("한 건", "두 건", "몇 개")도 건수 표현으로 본다.
# "조건"의 "건"은 앞에 숫자말이 없어 걸리지 않는다
_COUNT_WORDS = r"(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|몇|하나|둘|셋|넷)"
_KOREAN_COUNT = re.compile(rf"(?<![가-힣]){_COUNT_WORDS}\s*(?:건|개)")
# 숫자 바로 뒤에 건·개가 붙은 표현(3건, 13 개)은 조건 이름에 같은 숫자가 있어도 건수다
_DIGIT_COUNT = re.compile(r"\d+\s*(?:건|개)")


def is_valid(text: str, labels: Sequence[str] = ()) -> bool:
    """비어 있지 않고, 너무 길지 않고, 건수 숫자가 섞여 있지 않아야 한다.

    조건 이름에 원래 들어 있는 숫자(예: "3월", "2025년 가을")는 문장에 있어도 된다.
    """
    if not text or len(text) > MAX_LENGTH:
        return False
    if _DIGIT_COUNT.search(text) or _KOREAN_COUNT.search(text):
        return False
    allowed = set(re.findall(r"\d+", " ".join(labels)))
    return all(number in allowed for number in re.findall(r"\d+", text))
