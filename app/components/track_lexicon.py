"""그 사용자가 저장한 곡의 가수·곡명 사전 — 기능5 기억 검색 (순수 함수).

가수·곡명은 수가 무한해서 규칙 파일(question_rules.yaml)에 적을 수 없다. 대신 이 사용자의 자물쇠에
달린 곡의 가수·곡명을 읽어 와서, 질문 글자에 그 이름이 있는지 본다. 전 세계 가수 사전이 필요 없다.

- 가수: `artist_name`은 한 칸에 "A, B"처럼 여러 명이 들어 있을 수 있어서 한 명씩 나눠 비교한다.
- 곡명: 괄호·feat.·" - Remastered" 같은 꼬리를 뗀 이름도 비교한다 ("밤편지"로 "밤편지 (Lullaby)").
- 비교는 공백·대소문자·기호를 무시한다 ("RESCENE", "rescene", "Res Cene").
- 질문에서 이름을 찾을 때는 이름이 더 긴 낱말의 일부로 잡히지 않게 앞뒤를 본다 (name_matcher와
  같은 생각). 두 글자 미만의 이름과 "노래"·"음악" 같은 일반어는 사전에 넣지 않는다.

날씨·장소 같은 규칙이 먼저 글자를 가져가므로(question_parser), 여기서는 남은 글자에서만 찾는다.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache

from app.components.name_matcher import blank_out
from app.db.repositories.memory_record_repository import TrackName

MIN_NAME_LENGTH = 2

# 곡명·가수명으로 잡히면 안 되는 일반어 (질문의 뼈대 말)
STOPWORDS = frozenset(
    {"노래", "음악", "곡", "가수", "앨범", "플레이리스트", "그날", "오늘", "어제", "요즘", "최근"}
)

_ARTIST_SPLIT = re.compile(r"\s*[,/&]\s*")
_BRACKETS = re.compile(r"\s*[\(\[\{（【].*?[\)\]\}）】]")
_DASH_SUFFIX = re.compile(r"\s+-\s+.*$")

# 이름 바로 뒤에 올 수 있는 말. name_matcher.FOLLOWING에 "밤편지라는 노래" 같은 말을 더했다
_FOLLOWING = (
    r"(?=$|[^가-힣]|에서|에|의|은|는|이|가|을|를|도|와|과|랑|쪽|"
    r"라는|이라는|란|이란|라고|이라고|노래|곡|음악|앨범)"
)
# 이름임을 분명히 말하는 꼴: 따옴표·괄호로 감싸거나 "X라는 노래"처럼 말한다
_OPEN_QUOTES = "'\"‘“「『《〈"
_CLOSE_QUOTES = "'\"’”」』》〉"
_CONNECTIVE = re.compile(r"\s*(?:이라는|라는|이란|이라고\s*하는|라고\s*하는)")
_NOT_AFTER_WORD = r"(?<![0-9A-Za-z가-힣])"
_NOT_BEFORE_LATIN = r"(?![0-9A-Za-z])"


@lru_cache(maxsize=8192)
def compact(text: str) -> str:
    """비교용: 글자와 숫자만 남기고 소문자로 맞춘다 (공백·기호 무시)."""
    return "".join(re.findall(r"[^\W_]", text.lower()))


def artist_names(artist_name: str | None) -> list[str]:
    """ "A, B" 같은 가수 칸을 한 명씩 나눈다."""
    if not artist_name:
        return []
    return [part.strip() for part in _ARTIST_SPLIT.split(artist_name) if part.strip()]


def title_variants(title: str | None) -> list[str]:
    """곡명과, 괄호·꼬리를 뗀 곡명 (같으면 하나만)."""
    if not title or not title.strip():
        return []
    base = title.strip()
    stripped = _DASH_SUFFIX.sub("", _BRACKETS.sub("", base)).strip()
    return [base] if not stripped or stripped == base else [base, stripped]


def artist_keys(artist_name: str | None) -> frozenset[str]:
    """점수 계산용: 이 곡의 가수들을 비교용 글자로."""
    return frozenset(k for k in (compact(a) for a in artist_names(artist_name)) if k)


def title_keys(title: str | None) -> frozenset[str]:
    return frozenset(k for k in (compact(t) for t in title_variants(title)) if k)


def artist_name_keys(tracks: Iterable[TrackName]) -> set[str]:
    """이 곡들의 가수(한 명씩)를 비교용 글자로 — 가수 별칭 표를 찾는 열쇠."""
    return {key for t in tracks for a in artist_names(t.artist_name) if (key := compact(a))}


def _usable(key: str) -> bool:
    return len(key) >= MIN_NAME_LENGTH and key not in STOPWORDS


def _pattern(key: str) -> re.Pattern[str]:
    # 글자 사이에는 공백·기호가 있어도 된다 ("Hype Boy", "hype-boy")
    body = r"[\W_]*".join(re.escape(c) for c in key)
    return re.compile(_NOT_AFTER_WORD + body + _NOT_BEFORE_LATIN + _FOLLOWING, re.IGNORECASE)


@dataclass(frozen=True)
class NameMatch:
    """질문에서 찾은 가수·곡명 하나와 그 자리."""

    kind: str  # "artist" | "title"
    name: str  # 안내 문장에 쓸 이름
    start: int  # 질문 글자에서의 자리 (따옴표·"라는"까지 포함)
    end: int
    explicit: bool  # 따옴표로 감쌌거나 "X라는 노래"처럼 이름이라고 분명히 말했는지


class TrackLexicon:
    """이 사용자가 저장한 곡의 가수·곡명 → 질문에서 찾아 주는 사전."""

    def __init__(self, artists: Mapping[str, str], titles: Mapping[str, str]):
        # 비교용 글자 → 안내 문장에 쓸 이름
        self._artists = dict(artists)
        self._titles = dict(titles)

    @classmethod
    def build(
        cls, tracks: Iterable[TrackName], aliases: Mapping[str, Sequence[str]] | None = None
    ) -> "TrackLexicon":
        """aliases: 가수 비교용 글자(compact) → 다른 표기들 ("rescene" → ["리센느"]).

        다른 표기로 질문해도 저장된 이름의 가수로 찾는다. 안내 문장에는 저장된 이름을 쓴다."""
        aliases = aliases or {}
        artists: dict[str, str] = {}
        titles: dict[str, str] = {}
        for track in tracks:
            for name in artist_names(track.artist_name):
                key = compact(name)
                if _usable(key):
                    artists.setdefault(key, name)
                    for alias in aliases.get(key, ()):
                        alias_key = compact(alias)
                        if _usable(alias_key):
                            artists.setdefault(alias_key, name)
            variants = title_variants(track.title)
            if variants:
                display = variants[-1]  # 꼬리를 뗀 이름이 안내 문장에 쓰기 좋다
                for variant in variants:
                    key = compact(variant)
                    if _usable(key):
                        titles.setdefault(key, display)
        return cls(artists, titles)

    def __bool__(self) -> bool:
        return bool(self._artists or self._titles)

    def matches(self, text: str) -> list[NameMatch]:
        """질문에서 사전의 가수·곡명이 있는 자리를 모두 찾는다 (글자는 지우지 않는다)."""
        found = [
            *_spans(text, self._artists, "artist"),
            *_spans(text, self._titles, "title"),
        ]
        return sorted(found, key=lambda m: (m.start, -m.end))

    def find_artists(self, text: str) -> tuple[list[str], str]:
        return _find(text, self._artists)

    def find_titles(self, text: str) -> tuple[list[str], str]:
        return _find(text, self._titles)


def _spans(text: str, entries: Mapping[str, str], kind: str) -> list[NameMatch]:
    """_find와 같은 규칙으로 이름의 자리를 찾는다. 글자는 지우지 않고, 자리는 서로 겹치지 않는다."""
    taken: list[tuple[int, int]] = []
    found: list[NameMatch] = []
    for key, display in sorted(entries.items(), key=lambda item: -len(item[0])):  # 긴 이름부터
        if key not in compact(text):
            continue
        for match in _pattern(key).finditer(text):
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue
            taken.append((match.start(), match.end()))
            start, end, explicit = _mark(text, match.start(), match.end())
            found.append(NameMatch(kind, display, start, end, explicit))
    return found


def _mark(text: str, start: int, end: int) -> tuple[int, int, bool]:
    """따옴표로 감쌌거나 "X라는"이 뒤따르면 그 부분까지 자리를 넓히고 분명한 이름으로 본다."""
    left = start
    while left > 0 and text[left - 1].isspace():
        left -= 1
    right = end
    while right < len(text) and text[right].isspace():
        right += 1
    if (
        left > 0
        and text[left - 1] in _OPEN_QUOTES
        and right < len(text)
        and text[right] in _CLOSE_QUOTES
    ):
        return left - 1, right + 1, True
    connective = _CONNECTIVE.match(text, end)
    if connective:
        return start, connective.end(), True
    return start, end, False


def _find(text: str, entries: Mapping[str, str]) -> tuple[list[str], str]:
    """질문에서 사전의 이름을 모두 찾는다. (찾은 이름들, 찾은 글자를 공백으로 지운 질문)."""
    found: list[str] = []
    # 긴 이름부터: 더 긴 이름이 먼저 잡혀야 그 안의 짧은 이름이 또 잡히지 않는다
    for key, display in sorted(entries.items(), key=lambda item: -len(item[0])):
        if key not in compact(text):  # 안 들어 있는 이름은 정규식을 만들지 않는다
            continue
        while (match := _pattern(key).search(text)) is not None:
            text = blank_out(text, match.start(), match.end())
            if display not in found:
                found.append(display)
    return found, text
