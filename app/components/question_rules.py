"""질문 해석 규칙 읽기 — 기능5 기억 검색. app/question_rules.yaml을 읽고 검증해 패턴으로 만든다.

규칙 단어는 코드에 박지 않고 yaml에 둔다. 단어를 고치거나 늘릴 때 코드를 안 건드리려는 것이다.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from app.components.music_tags import MOOD_VOCAB
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
class QuestionRules:
    seasons: tuple[SeasonRule, ...]
    weather: tuple[WeatherRule, ...]
    temperature: tuple[RangeRule, ...]
    mood: tuple[RangeRule, ...]
    song_moods: tuple[SongMoodRule, ...]

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
        )


def rule_pattern(word: str) -> re.Pattern[str]:
    """규칙 단어 하나를 패턴으로 만든다. 공백은 있어도 없어도 되고, 앞에 한글이 붙으면 안 된다."""
    chars = [re.escape(c) for c in word if not c.isspace()]
    if not chars:
        raise ValueError("규칙 단어가 비어 있어요.")
    return re.compile(NOT_AFTER_HANGUL + r"\s*".join(chars))


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
