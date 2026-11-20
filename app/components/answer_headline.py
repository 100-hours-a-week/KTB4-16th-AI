"""기억 검색 답변 앞머리 문장 — LLM이 조건 이름과 찾은 자물쇠의 값을 보고 쓴다 (RAG).

자물쇠 목록과 건수는 프론트가 records로 따로 보여주므로 문장은 "어떤 조건으로 찾았는지, 찾았는지"를
말하고, 찾은 자물쇠의 값을 한두 가지 짚는다. **건수 숫자는 쓰지 않는다.** 말투는 따뜻하고 감성적인
존댓말이다(memory-headline-v4, 조건만 받는 v3는 Claude 폴백용 SYSTEM_PROMPT로 남아 있다).

LLM이 받는 것:
- 질문에서 규칙이 읽어 낸 조건 이름(예: "홍대 근처", "비 오는 날")과 결과 유무(있음·없음·많아서
  일부만). 가수·곡명 조건은 이름 대신 종류만 보낸다(부른 쪽에서 가린다).
- 조건에 모두 맞는 자물쇠 중 위에서부터 몇 건의 값: 월·동네·날씨·곡명·가수. 카드에 나오는 값과
  같은 값이다. **코멘트·이미지·질문 원문은 보내지 않는다.**

기록 값은 RunPod의 로컬 Gemma만 받는다(HeadlineWriter의 grounded). Claude로 넘어가면 기록 값 없이
조건 이름과 결과 유무만 보낸다. 그래서 기록 값이 외부 서비스로 나가지 않는다.

LLM이 값을 지어낼 수 있어서 글을 검사한다. 건수 숫자, 받지 않은 월·동네·날씨, 받지 않은
곡명(따옴표로 감싼 말), 곡명과 가수의 잘못된 짝(다른 기록의 가수를 붙임)이 섞이면 None을
돌려주고, 부른 쪽이 템플릿 문장(memory_answer.template_headline)으로 대신한다.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from langsmith import traceable

from app.clients.llm_client import LLMClient, LLMError
from app.db.repositories.memory_record_repository import ConditionRow

logger = logging.getLogger("muro.llm")

PROMPT_VERSION = "memory-headline-v4"

MAX_LENGTH = 200  # 앞머리는 1~2문장이다. 이보다 길면 목록을 풀어 쓰는 등 지시를 어긴 것으로 본다
MAX_RECORDS = 3  # LLM에 값을 보내는 자물쇠 수 (위에서부터)
MAX_VALUE_LENGTH = 60  # 곡명·가수·동네 값 하나의 최대 길이 (너무 긴 값은 잘라서 보낸다)

# 조건 이름과 결과 유무만 받을 때의 프롬프트(v3). 기록 값이 없을 때와 Claude 폴백이 쓴다
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

# 찾은 자물쇠의 값까지 받을 때의 프롬프트(v4). 로컬 Gemma만 쓴다
RAG_SYSTEM_PROMPT = """너는 음악 기록 앱 '뮤로'의 검색 결과 안내 문장을 쓴다.
사용자가 저장해 둔 기록을 '자물쇠'라고 부른다. 이 문장은 사용자가 남겨 둔 기억을 다시 만나게 해 주는
첫마디다. 질문한 조건과, 그 조건에 맞게 찾은 자물쇠 몇 개의 값(월·동네·날씨·곡·가수)을 받는다.

말투:
- 따뜻하고 감성적인 존댓말 1~2문장으로 쓴다. 그날을 떠올리게 하는 부드러운 말을 한마디 곁들인다.
- 과장하거나 시처럼 길게 쓰지 않는다. 담백한 감성이면 충분하다.

규칙:
- 질문한 조건 이름들을 자연스러운 말로 이어서 말하고, 조건에 맞는 자물쇠를 찾았다고 알린다.
- 받은 자물쇠의 값 중 한두 가지(예: 동네, 곡)를 짚어서 말한다. 값은 받은 그대로 쓴다.
  곡명은 받은 그대로 작은따옴표로 감싸고, 고치거나 번역하지 않는다.
- 곡명과 가수는 같은 줄에 있는 것끼리만 함께 말한다. 다른 줄의 곡과 가수를 섞지 않는다.
- 받은 값에 없는 내용(곡, 가수, 동네, 월, 날씨, 분위기, 사람)을 지어내지 않는다.
  감정을 담는 것은 괜찮지만 사실을 더하지는 않는다.
- 자물쇠를 하나하나 나열하지 않는다. 가장 위의 자물쇠 하나나 여러 자물쇠의 공통점만 말한다.
- **건수는 말하지 않는다.** 몇 건, 몇 개, 한 건, 두 건 같은 숫자 표현을 쓰지 않는다.
  숫자는 받은 월(예: 10월)이나 조건 이름에 있는 것만 그대로 쓴다.
- "많아서 일부만 보여줌"이라고 받으면 조건에 맞는 자물쇠가 많아서 일부만 보여드린다고 알린다.
- 일부만 맞는 자물쇠가 있다고 알려 주면 마지막에 비슷한 자물쇠도 함께 보여준다고 짧게 덧붙인다.
- 곡명 말고는 따옴표를 쓰지 않는다. 이모지 없이 문장만 출력한다."""

# 날씨 코드(records.weather_condition)의 한국어 표현. 규칙 파일(question_rules)의 코드와 같다
WEATHER_TEXT = {
    "CLEAR": "맑음",
    "CLOUDY": "구름 많음",
    "OVERCAST": "흐림",
    "RAIN": "비",
    "SNOW": "눈",
    "RAIN_SNOW": "비와 눈",
    "SHOWER": "소나기",
}


@dataclass(frozen=True)
class RecordFacts:
    """찾은 자물쇠 하나의 값 — 카드에 나오는 값과 같다. 없는 값은 None."""

    month: int | None = None
    dong: str | None = None
    weather_code: str | None = None  # WEATHER_TEXT의 키
    title: str | None = None
    artist: str | None = None


def facts_from_row(row: ConditionRow) -> RecordFacts:
    """조건 점수용으로 읽어 둔 자물쇠 행에서 LLM에 보낼 값만 뽑는다 (코멘트·좌표·기온은 뺀다)."""
    code = (row.weather_condition or "").strip().upper()
    return RecordFacts(
        month=row.created_at.month,
        dong=_clean(row.dong_name),
        weather_code=code if code in WEATHER_TEXT else None,
        title=_clean(row.track_title),
        artist=_clean(row.artist_name),
    )


def _clean(value: str | None) -> str | None:
    """프롬프트에 넣을 값을 한 줄로 정리한다 (줄바꿈으로 지시문을 끼우지 못하게, 길면 자른다)."""
    if not value:
        return None
    return " ".join(value.split())[:MAX_VALUE_LENGTH] or None


class HeadlineWriter:
    """앞머리 문장을 쓴다.

    llm: 먼저 부르는 LLM. grounded가 True면 이 LLM에만 찾은 자물쇠의 값을 보낸다(RunPod의 Gemma).
    fallback: llm 호출이 LLMError로 실패하면 부르는 LLM(Claude). 기록 값 없이 조건 이름만 보낸다.
    """

    def __init__(
        self, llm: LLMClient, *, fallback: LLMClient | None = None, grounded: bool = False
    ):
        self._llm = llm
        self._fallback = fallback
        self._grounded = grounded

    @property
    def model(self) -> str:
        return self._llm.model

    @traceable(name="headline_llm", run_type="llm")
    async def write(
        self,
        labels: Sequence[str],
        *,
        found: int,
        total: int,
        suggestions: int,
        records: Sequence[RecordFacts] = (),
    ) -> str | None:
        """앞머리 문장. 쓸 수 없는 응답이면 None.

        labels: 조건 이름들. found·total·suggestions는 결과가 있는지, 많아서 일부만 보내는지,
        제안이 있는지를 정하는 데만 쓰고 숫자는 LLM에 보내지 않는다.
        records: 조건에 모두 맞는 자물쇠의 값. 이 글쓴이가 grounded일 때만 LLM에 보낸다.
        """
        shown = list(records)[:MAX_RECORDS] if self._grounded else []
        try:
            response = await self._complete(self._llm, labels, found, total, suggestions, shown)
        except LLMError as e:
            if self._fallback is None:
                raise
            logger.warning(
                "기억 검색 답변 LLM(%s) 실패, 대체 LLM(%s)으로 넘겨요(기록 값 없이): %s",
                self._llm.model,
                self._fallback.model,
                e,
            )
            shown = []
            response = await self._complete(self._fallback, labels, found, total, suggestions, [])
        text = _unwrap(response.strip())
        return text if is_valid(text, labels, shown) else None

    @staticmethod
    async def _complete(
        llm: LLMClient,
        labels: Sequence[str],
        found: int,
        total: int,
        suggestions: int,
        records: Sequence[RecordFacts],
    ) -> str:
        return await llm.complete(
            system=RAG_SYSTEM_PROMPT if records else SYSTEM_PROMPT,
            prompt=build_prompt(
                labels, found=found, total=total, suggestions=suggestions, records=records
            ),
            max_tokens=200,
        )


def build_prompt(
    labels: Sequence[str],
    *,
    found: int,
    total: int,
    suggestions: int,
    records: Sequence[RecordFacts] = (),
) -> str:
    lines = [f"질문한 조건: {', '.join(labels)}"]
    if found == 0:
        lines.append("조건에 모두 맞는 자물쇠: 없음")
    elif total > found:
        lines.append("조건에 모두 맞는 자물쇠: 많아서 일부만 보여줌")
    else:
        lines.append("조건에 모두 맞는 자물쇠: 있음")
    if suggestions:
        lines.append("일부만 맞는 자물쇠: 있음 (화면에 따로 보여줌)")
    facts = [line for line in map(_facts_line, list(records)[:MAX_RECORDS]) if line]
    if facts:
        lines.append("조건에 맞게 찾은 자물쇠의 값 (위에서부터 순서대로):")
        lines += facts
    return "\n".join(lines)


def _facts_line(facts: RecordFacts) -> str:
    parts = []
    if facts.month:
        parts.append(f"{facts.month}월")
    if facts.dong:
        parts.append(f"동네 {facts.dong}")
    weather = WEATHER_TEXT.get(facts.weather_code or "")
    if weather:
        parts.append(f"날씨 {weather}")
    if facts.title:
        parts.append(f"곡 '{facts.title}'")
    if facts.artist:
        parts.append(f"가수 {facts.artist}")
    return "- " + " / ".join(parts) if parts else ""


def _unwrap(text: str) -> str:
    """문장 전체를 큰따옴표로 감싸 보냈으면 벗긴다 (곡명은 작은따옴표라 건드리지 않는다)."""
    if len(text) > 1 and text[0] in '"“' and text[-1] in '"”':
        return text[1:-1].strip()
    return text


# 한국어로 쓴 개수("한 건", "두 건", "몇 개")도 건수 표현으로 본다.
# "조건"의 "건"은 앞에 숫자말이 없어 걸리지 않는다
_COUNT_WORDS = r"(?:한|두|세|네|다섯|여섯|일곱|여덟|아홉|열|몇|하나|둘|셋|넷)"
_KOREAN_COUNT = re.compile(rf"(?<![가-힣]){_COUNT_WORDS}\s*(?:건|개)")
# 숫자 바로 뒤에 건·개가 붙은 표현(3건, 13 개)은 조건 이름에 같은 숫자가 있어도 건수다
_DIGIT_COUNT = re.compile(r"\d+\s*(?:건|개)")

# 동 이름 같은 말: 한글 두 글자 이상 + 동 ("합정동", "합정동에서"). "동안"은 뺀다
_DONG = re.compile(r"[가-힣]{2,}동(?!안)")
# 따옴표로 감싼 말
_QUOTED = re.compile(r"['‘“\"][^'‘’“”\"]+['’”\"]")

# 글에서 날씨를 말하는 표현 (날씨 코드별). 받지 않은 날씨를 지어냈는지 보는 데 쓴다
_WEATHER_WORDS = {
    "CLEAR": re.compile(r"맑[은았음게]"),
    "CLOUDY": re.compile(r"구름"),
    "OVERCAST": re.compile(r"흐[린렸림]"),
    "RAIN": re.compile(r"(?<![가-힣])비(?:가|를|는|도|와|에|\s*오|\s*내)|빗[속줄방소]|장마"),
    "SNOW": re.compile(r"(?<![가-힣])눈(?:이|을|은|도|과|에|\s*오|\s*내)|함박눈|첫눈|눈발"),
    "RAIN_SNOW": re.compile(r"진눈깨비"),
    "SHOWER": re.compile(r"소나기"),
}
# 받은 날씨와 함께 말해도 되는 비슷한 날씨
_WEATHER_ALSO_OK = {
    "RAIN_SNOW": {"RAIN", "SNOW"},
    "SHOWER": {"RAIN"},
    "CLOUDY": {"OVERCAST"},
    "OVERCAST": {"CLOUDY"},
}


def is_valid(text: str, labels: Sequence[str] = (), records: Sequence[RecordFacts] = ()) -> bool:
    """비어 있지 않고, 너무 길지 않고, 건수 숫자가 섞여 있지 않아야 한다.

    조건 이름에 원래 들어 있는 숫자(예: "3월", "2025년 가을")는 문장에 있어도 된다.
    records가 있으면(LLM이 기록 값을 받았으면) 받은 월·동네의 숫자도 허용하고, 받지 않은 월·동네·
    날씨·곡명이 섞였는지도 본다.
    """
    if not text or len(text) > MAX_LENGTH:
        return False
    if _DIGIT_COUNT.search(text) or _KOREAN_COUNT.search(text):
        return False
    allowed = set(re.findall(r"\d+", " ".join(labels)))
    for facts in records:
        allowed |= set(re.findall(r"\d+", f"{facts.month or ''}월 {facts.dong or ''}"))
    if not all(number in allowed for number in re.findall(r"\d+", text)):
        return False
    return not records or _is_grounded(text, labels, records)


def _is_grounded(text: str, labels: Sequence[str], records: Sequence[RecordFacts]) -> bool:
    """글에 나온 곡명·가수·동네·날씨가 LLM이 받은 값(또는 조건 이름)에 있는 것뿐인지 본다."""
    if not _pairs_are_consistent(text, records):  # 곡명에 다른 기록의 가수를 붙였다
        return False

    # 받은 곡명·가수·조건 이름은 글에서 지우고(따옴표째) 남은 글로 검사한다.
    # 곡명 자체에 들어 있는 말("비 오는 날" 같은 제목)이 동네·날씨 검사에 걸리지 않게 하려는 것이다
    known = {f.title for f in records} | {f.artist for f in records} | set(labels)
    rest = text
    for name in sorted((n for n in known if n), key=len, reverse=True):
        rest = re.sub(rf"['‘“\"]?{re.escape(name)}['’”\"]?", " ", rest, flags=re.IGNORECASE)
    if _QUOTED.search(rest):  # 받지 않은 곡명을 따옴표로 감쌌다
        return False

    allowed_dongs = " ".join([*(f.dong or "" for f in records), *labels])
    if any(token not in allowed_dongs for token in _DONG.findall(rest)):
        return False

    allowed_weather = _weather_in(" ".join(labels))
    for facts in records:
        code = facts.weather_code
        if code:
            allowed_weather |= {code, *_WEATHER_ALSO_OK.get(code, ())}
    return _weather_in(rest) <= allowed_weather


# 가수 칸에 여러 명이 적힌 것을 낱개로 나누는 구분 ("A, B", "A & B", "A feat. B")
_ARTIST_SPLIT = re.compile(r"\s*(?:,|&|/|\bfeat\.?)\s*", re.IGNORECASE)


def _artist_names(artist: str | None) -> set[str]:
    """가수 칸의 이름들을 낱개로 (소문자). 한 글자 이름은 글 속 다른 말과 구분할 수 없어 뺀다."""
    if not artist:
        return set()
    return {n.strip().lower() for n in _ARTIST_SPLIT.split(artist) if len(n.strip()) >= 2}


def _pairs_are_consistent(text: str, records: Sequence[RecordFacts]) -> bool:
    """글에 곡명과 가수가 함께 나오면, 그 가수가 말한 곡의 가수여야 한다.

    A 기록의 곡명에 B 기록의 가수를 붙이면 거절한다. 곡명 없이 가수만 나오거나 가수 없이 곡명만
    나오면 짝이 없으니 보지 않는다. 받지 않은 가수 이름을 지어낸 것은 여기서 잡지 못한다.
    """
    lowered = text.lower()
    titled = [f for f in records if f.title and f.title.lower() in lowered]
    if not titled:
        return True
    # 곡명 안에 가수 이름이 들어 있어도(제목이 가수 이름인 경우) 가수를 말한 것으로 보지 않는다
    without_titles = lowered
    for facts in titled:
        without_titles = without_titles.replace(facts.title.lower(), " ")
    said_by = [_artist_names(f.artist) for f in titled]
    for facts in records:
        names = _artist_names(facts.artist)
        if any(name in without_titles for name in names) and not any(names & s for s in said_by):
            return False
    return True


def _weather_in(text: str) -> set[str]:
    return {code for code, pattern in _WEATHER_WORDS.items() if pattern.search(text)}
