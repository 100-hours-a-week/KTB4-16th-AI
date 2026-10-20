"""가수 이름 → 다른 표기(영어↔한글) — 기억 검색의 가수 조건용.

스포티파이에는 한국 가수도 영어로("RESCENE"), 일부는 한글로("아이유") 저장돼 있고, 사용자는 한글로도
영어로도 묻는다. 저장된 이름이 한글이면 영어 표기를, 아니면 한글 표기를 LLM에게 묻는다.

- 아는 가수는 실제로 쓰이는 표기를, 모르는 가수는 이름의 발음을 옮긴 표기 하나를 받는다.
  덜 알려진 가수는 LLM이 모른다고만 답해 별칭이 비는 일이 많았다 (개발 DB 이름 시험에서 영어→한글
  9명 중 5명). 별칭은 이 사용자가 저장한 가수와 질문 글자를 맞춰 보는 데만 쓰여서, 발음으로 옮긴
  표기가 틀려도 질문에 그 글자가 있을 때만 영향이 있다.
- 가수 이름이 아닌 말("Various Artists" 같은)은 "없음"으로 답하게 한다 (빈 결과).
- 답이 저장된 이름과 같은 문자(언어)로 오면 버린다 (이름을 그대로 되돌려 주는 답 방지).
- 형식을 못 알아듣는 답은 오류로 낸다. 호출한 쪽이 저장하지 않고 다음에 다시 묻는다.
"""

import re

from app.clients.llm_client import LLMClient
from app.components.track_lexicon import compact

PROMPT_VERSION = "artist-alias-v1"

MAX_ALIASES = 3
MAX_ALIAS_LENGTH = 60

_TO_KOREAN = """너는 음악 가수 이름 표기 도우미다.
스포티파이에 저장된 가수 이름을 보고, 한국 사람이 그 가수를 한글로 검색할 때 쓰는 표기를 알려 준다.

형식(정확히 이대로, 다른 말 붙이지 않는다):
표기: <한글 표기 1~3개, 쉼표로. 가장 흔한 표기를 앞에>

규칙:
- 아는 가수는 한국에서 통용되는 표기를 쓴다. 한국어 표기가 둘 이상 널리 쓰이면(예: 비틀즈, 비틀스)
  둘 다 쓴다.
- 모르는 가수는 이름의 발음을 한글로 옮긴 표기를 가장 그럴듯한 순서로 2~3개 쓴다 (예: 받침과
  어말 "느"·"스" 같은 쓰임이 다른 표기를 함께). 한국 사람 이름을 로마자로 쓴 것이면 흔한 한글
  이름으로 되돌린다 (예: Moon Sung Jae → 문성재).
- 가수 이름이 아닌 말("Various Artists" 같은)은 "표기: 없음"이라고만 쓴다."""

_TO_ENGLISH = """너는 음악 가수 이름 표기 도우미다.
스포티파이에 한글로 저장된 가수 이름을 보고, 그 가수가 스포티파이·유튜브 등에서 실제로 쓰는
영어(로마자) 표기를 알려 준다.

형식(정확히 이대로, 다른 말 붙이지 않는다):
표기: <영어 표기 1~3개, 쉼표로. 가장 흔한 표기를 앞에>

규칙:
- 아는 가수는 실제로 쓰이는 표기를 쓴다.
- 모르는 가수는 한국어 로마자 표기법에 따라 옮긴 표기를 가장 그럴듯한 순서로 1~2개 쓴다.
- 가수 이름이 아닌 말은 "표기: 없음"이라고만 쓴다."""

_HANGUL = re.compile(r"[가-힣]")
_LATIN = re.compile(r"[A-Za-z]")


def needs_alias(name: str) -> bool:
    """글자가 하나도 없는 이름("-", "...")은 물어볼 것이 없다."""
    return any(ch.isalpha() for ch in name)


class ArtistAliasTagger:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def version(self) -> str:
        """LLM이나 프롬프트가 바뀌면 달라진다 — 저장된 별칭을 다시 만들 때 비교한다."""
        return f"{self._llm.model}|{PROMPT_VERSION}"

    async def aliases(self, name: str) -> tuple[str, ...]:
        """다른 표기 목록. 가수 이름이 아닌 말은 빈 튜플. 응답 형식이 틀리면 ValueError."""
        to_english = bool(_HANGUL.search(name))
        text = await self._llm.complete(
            system=_TO_ENGLISH if to_english else _TO_KOREAN,
            prompt=f"가수: {name}",
            max_tokens=100,
        )
        return _parse(text, name, to_english=to_english)


def _parse(text: str, name: str, *, to_english: bool) -> tuple[str, ...]:
    line = next((raw.strip() for raw in text.splitlines() if raw.strip().startswith("표기:")), None)
    if line is None:
        raise ValueError("가수 별칭 응답에 '표기:' 줄이 없음")
    wanted = _LATIN if to_english else _HANGUL
    own = compact(name)
    found: list[str] = []
    for raw in line.removeprefix("표기:").split(","):
        alias = raw.strip().strip("'\"")
        if not alias or alias == "없음" or len(alias) > MAX_ALIAS_LENGTH:
            continue
        if not wanted.search(alias):
            continue
        key = compact(alias)
        if not key or key == own or any(compact(f) == key for f in found):
            continue
        found.append(alias)
    return tuple(found[:MAX_ALIASES])
