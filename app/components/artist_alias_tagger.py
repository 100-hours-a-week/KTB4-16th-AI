"""가수 이름 → 사용자가 이 가수를 가리킬 때 쓸 만한 다른 글자들 — 기억 검색의 가수 조건용.

스포티파이에는 한국 가수도 영어로("RESCENE"), 일부는 한글로("아이유") 저장돼 있고, 사용자는 한글로도
영어로도, 때로는 줄임말·별명이나 오타로 묻는다. 저장된 이름 하나마다 LLM에게 아래를 한 번에 받는다.

- 영어↔한글 표기, 같은 이름의 다른 철자("아일리시"·"아일리쉬"), 흔한 오타, 줄임말·별명·본명
  ("지디"·"권지용"). 질문할 때 LLM을 부르지 않고 여기서 미리 받아 둔다.
- 모르는 가수는 이름의 발음을 옮긴 표기를 받는다. 덜 알려진 가수는 LLM이 모른다고만 답해 별칭이
  비는 일이 많았다 (개발 DB 이름 시험에서 영어→한글 9명 중 5명). 별칭은 이 사용자가 저장한 가수와
  질문 글자를 맞춰 보는 데만 쓰여서, 틀린 표기도 질문에 그 글자가 있을 때만 영향이 있다.
- 가수 이름이 아닌 말("Various Artists" 같은)은 "없음"으로 답하게 한다 (빈 결과).
- 쓸모없는 답은 거른다: 저장된 이름과 같은 글자, 너무 짧은 것, "노래"·"밴드"·"멤버" 같은 군더더기가
  붙은 것 ("에드 시런 노래").
- 형식을 못 알아듣는 답은 오류로 낸다. 호출한 쪽이 저장하지 않고 다음에 다시 묻는다.
"""

from app.clients.llm_client import LLMClient
from app.components.track_lexicon import MIN_NAME_LENGTH, compact

PROMPT_VERSION = "artist-alias-v2"

MAX_ALIASES = 6
MAX_ALIAS_LENGTH = 60

# 가수 이름이 아니라 설명이 붙은 답을 거른다. 같은 묶음의 한글·영어 말 중 하나라도 저장된 이름에
# 들어 있으면 거르지 않는다 ("더 밴드 페리"와 "The Band Perry"의 "밴드"·"band")
_FILLER_WORDS = (
    ("노래",),
    ("곡",),
    ("밴드", "band"),
    ("그룹", "group"),
    ("멤버", "member"),
    ("가수", "singer"),
    ("님",),
)

SYSTEM_PROMPT = """너는 음악 가수 이름 표기 도우미다.
한국 사용자가 음악 기록을 검색할 때 이 가수를 가리키려고 쓸 만한 글자들을 알려 준다.
다음을 포함한다:
- 영어↔한글 표기 (저장된 이름이 영어면 한글 표기를, 한글이면 영어(로마자) 표기를 꼭 넣는다)
- 같은 이름의 다른 한글·영어 철자 (예: 아일리시/아일리쉬)
- 사람들이 흔히 내는 오타
- 널리 쓰는 줄임말·별명·본명 (예: BTS는 방탄, G-DRAGON은 지디·권지용)
모르는 가수는 이름의 발음을 옮긴 표기(영어 이름이면 한글로, 한글 이름이면 로마자로)를 가장 그럴듯한
순서로 2~3개 쓴다. 한국 사람 이름을 로마자로 쓴 것이면 흔한 한글 이름으로 되돌린다
(예: Moon Sung Jae → 문성재).

형식(정확히 이대로, 다른 말 붙이지 않는다):
표기: <쉼표로 구분한 목록, 최대 6개, 가장 흔한 것부터>

"노래"·"밴드"·"멤버" 같은 설명을 붙이지 않는다. 가수 이름이 아닌 말은 "표기: 없음"이라고만 쓴다."""


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
        text = await self._llm.complete(
            system=SYSTEM_PROMPT, prompt=f"가수: {name}", max_tokens=200
        )
        return _parse(text, name)


def _parse(text: str, name: str) -> tuple[str, ...]:
    line = next((raw.strip() for raw in text.splitlines() if raw.strip().startswith("표기:")), None)
    if line is None:
        raise ValueError("가수 별칭 응답에 '표기:' 줄이 없음")
    own = compact(name)
    found: list[str] = []
    for raw in line.removeprefix("표기:").split(","):
        alias = raw.strip().strip("'\"")
        if not alias or alias == "없음" or len(alias) > MAX_ALIAS_LENGTH:
            continue
        key = compact(alias)
        if len(key) < MIN_NAME_LENGTH or key == own or any(compact(f) == key for f in found):
            continue
        if _has_filler(alias, name):
            continue
        found.append(alias)
    return tuple(found[:MAX_ALIASES])


def _has_filler(alias: str, name: str) -> bool:
    alias, name = alias.lower(), name.lower()
    return any(
        any(w in alias for w in group) and not any(w in name for w in group)
        for group in _FILLER_WORDS
    )
