"""질문 해석 규칙 읽기 — 기능5 기억 검색. app/question_rules.yaml을 읽고 검증해 패턴으로 만든다.

규칙 단어는 코드에 박지 않고 yaml에 둔다. 단어를 고치거나 늘릴 때 코드를 안 건드리려는 것이다.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.components.music_tags import GENRE_VOCAB, MOOD_VOCAB
from app.components.name_matcher import NOT_AFTER_HANGUL

DEFAULT_RULES_PATH = Path(__file__).resolve().parent.parent / "question_rules.yaml"

# 백엔드 records.weather_condition(Record.WeatherCondition enum)이 가질 수 있는 값
WEATHER_VALUES = frozenset({"CLEAR", "CLOUDY", "OVERCAST", "RAIN", "SNOW", "RAIN_SNOW", "SHOWER"})


@dataclass(frozen=True)
class SeasonRule:
    name: str
    months: frozenset[int]
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class WeatherRule:
    label: str
    conditions: frozenset[str]
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class TimeRule:
    """하루 중 시간대. hours는 맞는 시각(0~23시)을 풀어 둔 값이다."""

    label: str
    hours: frozenset[int]
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class WeekdayRule:
    label: str
    days: frozenset[int]  # 월=0 … 일=6
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class RangeRule:
    """기온·기분처럼 숫자 범위로 비교하는 규칙."""

    label: str
    low: float
    high: float
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class SongMoodRule:
    tag: str
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class GenreRule:
    tag: str  # GENRE_VOCAB의 장르 이름
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class QuestionRules:
    seasons: tuple[SeasonRule, ...]
    weather: tuple[WeatherRule, ...]
    temperature: tuple[RangeRule, ...]
    mood: tuple[RangeRule, ...]
    song_moods: tuple[SongMoodRule, ...]
    # 시간대·요일은 선택이다 (규칙 파일에 없으면 그 조건은 해석하지 않는다)
    times: tuple[TimeRule, ...] = ()
    weekdays: tuple[WeekdayRule, ...] = ()
    genres: tuple[GenreRule, ...] = ()

    @property
    def season_months(self) -> dict[str, frozenset[int]]:
        """기간 처리가 "작년 가을"의 범위를 계산할 때 쓴다. 이름은 계절 기본 이름만."""
        return {rule.name: rule.months for rule in self.seasons}

    @classmethod
    def load(cls, path: Path = DEFAULT_RULES_PATH) -> "QuestionRules":
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            seasons=tuple(_seasons(data.get("seasons"), path)),
            weather=tuple(_weather(data.get("weather"), path)),
            temperature=tuple(_ranges(data.get("temperature"), "temperature", path)),
            mood=tuple(_ranges(data.get("mood"), "mood", path)),
            song_moods=tuple(_song_moods(data.get("song_moods"), path)),
            times=tuple(_times(data.get("time_of_day"), path)),
            weekdays=tuple(_weekdays(data.get("weekday"), path)),
            genres=tuple(_genres(data.get("genre"), path)),
        )


def rule_pattern(word: str) -> re.Pattern[str]:
    """규칙 단어 하나를 패턴으로 만든다. 공백은 있어도 없어도 되고, 앞에 한글이 붙으면 안 된다."""
    chars = [re.escape(c) for c in word if not c.isspace()]
    if not chars:
        raise ValueError("규칙 단어가 비어 있어요.")
    return re.compile(NOT_AFTER_HANGUL + r"\s*".join(chars), re.IGNORECASE)


def _patterns(words: object, where: str, path: Path) -> tuple[re.Pattern[str], ...]:
    if not isinstance(words, Sequence) or isinstance(words, str) or not words:
        raise ValueError(f"{path}: {where}의 words가 비어 있어요.")
    return tuple(rule_pattern(str(w)) for w in words)


def _seasons(raw: object, path: Path) -> list[SeasonRule]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: seasons가 없어요.")
    rules = []
    for name, body in raw.items():
        months = frozenset(int(m) for m in body.get("months") or [])
        if not months or not all(1 <= m <= 12 for m in months):
            raise ValueError(f"{path}: 계절 {name}의 months는 1~12 사이 값이어야 해요.")
        rules.append(
            SeasonRule(str(name), months, _patterns(body.get("words"), f"계절 {name}", path))
        )
    return rules


def _weather(raw: object, path: Path) -> list[WeatherRule]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: weather가 없어요.")
    rules = []
    for body in raw:
        label = str(body.get("label", "")).strip()
        conditions = frozenset(str(c) for c in body.get("conditions") or [])
        if not label or not conditions or not conditions <= WEATHER_VALUES:
            raise ValueError(
                f"{path}: 날씨 규칙 '{label}'의 conditions를 확인해 주세요: {conditions}"
            )
        rules.append(WeatherRule(label, conditions, _patterns(body.get("words"), label, path)))
    return rules


def hours_between(start: int, end: int) -> frozenset[int]:
    """start시 이상 end시 미만의 시각들. start가 end보다 크면 자정을 넘긴다.

    예: [21, 5] → 21~23시와 0~4시.
    """
    if start < end:
        return frozenset(range(start, end))
    return frozenset([*range(start, 24), *range(0, end)])


def _times(raw: object, path: Path) -> list[TimeRule]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: time_of_day가 비어 있어요.")
    rules = []
    for body in raw:
        label = str(body.get("label", "")).strip()
        bounds = body.get("hours")
        if not label or not (isinstance(bounds, list) and len(bounds) == 2):
            raise ValueError(
                f"{path}: time_of_day 규칙 '{label}'에 label과 hours [시작, 끝]이 필요해요."
            )
        start, end = int(bounds[0]), int(bounds[1])
        if not (0 <= start <= 23 and 1 <= end <= 24 and start != end):
            raise ValueError(
                f"{path}: time_of_day 규칙 '{label}'의 hours는 시작 0~23, 끝 1~24이고 "
                "서로 달라야 해요."
            )
        rules.append(
            TimeRule(label, hours_between(start, end), _patterns(body.get("words"), label, path))
        )
    return rules


def _weekdays(raw: object, path: Path) -> list[WeekdayRule]:
    if raw is None:
        return []
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: weekday가 비어 있어요.")
    rules = []
    for body in raw:
        label = str(body.get("label", "")).strip()
        days = frozenset(int(d) for d in body.get("days") or [])
        if not label or not days or not all(0 <= d <= 6 for d in days):
            raise ValueError(f"{path}: weekday 규칙 '{label}'의 days는 0(월)~6(일) 값이어야 해요.")
        rules.append(WeekdayRule(label, days, _patterns(body.get("words"), label, path)))
    return rules


def _ranges(raw: object, section: str, path: Path) -> list[RangeRule]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: {section}가 없어요.")
    rules = []
    for body in raw:
        label = str(body.get("label", "")).strip()
        bounds = body.get("range")
        if not label or not (isinstance(bounds, list) and len(bounds) == 2):
            raise ValueError(
                f"{path}: {section} 규칙 '{label}'에 label과 range [최소, 최대]가 필요해요."
            )
        low, high = float(bounds[0]), float(bounds[1])
        if low > high:
            raise ValueError(
                f"{path}: {section} 규칙 '{label}'의 range는 최소가 최대보다 작아야 해요."
            )
        rules.append(RangeRule(label, low, high, _patterns(body.get("words"), label, path)))
    return rules


def _genres(raw: object, path: Path) -> list[GenreRule]:
    if raw is None:
        return []
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: genre가 비어 있어요.")
    unknown = set(raw) - set(GENRE_VOCAB)
    if unknown:
        raise ValueError(f"{path}: 곡 장르 목록에 없는 이름이에요: {sorted(unknown)}")
    return [
        GenreRule(str(tag), _patterns(words, f"장르 {tag}", path)) for tag, words in raw.items()
    ]


def _song_moods(raw: object, path: Path) -> list[SongMoodRule]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: song_moods가 없어요.")
    unknown = set(raw) - set(MOOD_VOCAB)
    if unknown:
        raise ValueError(f"{path}: 곡 무드 태그에 없는 이름이에요: {sorted(unknown)}")
    return [
        SongMoodRule(str(tag), _patterns(words, f"곡 무드 {tag}", path))
        for tag, words in raw.items()
    ]
