"""질문 해석 — 기능5 기억 검색. 질문 글자에서 "무엇을 찾는 질문인지"를 조건 목록으로 뽑는다.

DB를 부르지 않고 질문 글자와 우리 서버의 규칙 파일(places.yaml, regions.yaml,
question_rules.yaml)만 본다. 자물쇠에 저장된 값과 맞는지는 뒤 단계(condition_scoring)가 비교한다.

한 번 잡힌 글자는 공백으로 지워서 다른 규칙이 다시 쓰지 못하게 한다. 그래서 순서가 중요하다:
기간 → 장소 → 지역 → 동 → 계절·달 → 날씨 → 기온 → 기분 → 곡 무드.
(예: "서울숲"을 장소로 가져가야 "서울"이 지역으로 또 잡히지 않고, "우울했을 때"를 기분으로
가져가야 "우울"이 곡 무드로 또 잡히지 않는다.)
"""

import re
from collections.abc import Sequence
from datetime import date

from app.components.name_matcher import FOLLOWING, NOT_AFTER_HANGUL, blank_out
from app.components.period_parser import parse_periods
from app.components.place_dictionary import PlaceDictionary
from app.components.query_conditions import Condition
from app.components.question_rules import QuestionRules
from app.components.region_table import RegionTable

# "연남동", "성수동1가", "종로3가" 모양. 실제 동인지는 여기서 모른다 — "운동을", "감동이"도 잡히고,
# 이 사용자의 기록에 있는 동인지는 뒤 단계(resolve_conditions)가 확인해서 걸러낸다.
_DONG = re.compile(NOT_AFTER_HANGUL + r"([가-힣]{1,6}동(?:\d가)?|[가-힣]{1,6}\d가)" + FOLLOWING)
_MONTH = re.compile(r"(?<![0-9])(\d{1,2})\s*월")


class QuestionParser:
    def __init__(self, places: PlaceDictionary, regions: RegionTable, rules: QuestionRules):
        self._places = places
        self._regions = regions
        self._rules = rules

    @classmethod
    def load_default(cls) -> "QuestionParser":
        return cls(PlaceDictionary.load(), RegionTable.load(), QuestionRules.load())

    def parse(self, text: str, today: date) -> list[Condition]:
        """질문에서 조건들을 뽑는다. 조건이 없으면 빈 목록. today는 "지난달"의 기준일."""
        conditions: list[Condition] = []

        periods, text = parse_periods(text, today, self._rules.season_months)
        if periods:
            conditions.append(
                Condition(
                    "period",
                    "·".join(p.label for p in periods),
                    ranges=tuple((p.start, p.end) for p in periods),
                )
            )

        places, text = self._places.find(text)
        if places:
            names = "·".join(p.name for p in places)
            conditions.append(Condition("place", f"{names} 근처", places=tuple(places)))

        regions, text = self._regions.find(text)
        for hit in regions:
            conditions.append(
                Condition("region", hit.text, prefixes=hit.prefixes, reliable=hit.reliable)
            )

        dongs, text = _find_dongs(text)
        if dongs:
            conditions.append(Condition("dong", "·".join(dongs), names=tuple(dongs)))

        text = self._add_seasons(text, conditions)
        text = self._add_weather(text, conditions)
        text = self._add_ranges(text, self._rules.temperature, "temperature", conditions)
        text = self._add_ranges(text, self._rules.mood, "mood", conditions)
        self._add_song_moods(text, conditions)
        return conditions

    def _add_seasons(self, text: str, conditions: list[Condition]) -> str:
        names: list[str] = []
        months: set[int] = set()
        for rule in self._rules.seasons:
            matched, text = _consume(text, rule.patterns)
            if matched:
                names.append(rule.name)
                months |= rule.months
        if names:
            conditions.append(Condition("season", "·".join(names), months=frozenset(months)))

        # "10월에"처럼 해 없이 달만 말한 경우 (해까지 말한 "작년 10월"은 기간에서 이미 처리됨)
        month_numbers = [int(m) for m in _MONTH.findall(text) if 1 <= int(m) <= 12]
        if month_numbers:
            text = _MONTH.sub(lambda m: " " * len(m.group(0)), text)
            label = "·".join(f"{m}월" for m in month_numbers)
            conditions.append(Condition("season", label, months=frozenset(month_numbers)))
        return text

    def _add_weather(self, text: str, conditions: list[Condition]) -> str:
        labels: list[str] = []
        values: set[str] = set()
        for rule in self._rules.weather:
            matched, text = _consume(text, rule.patterns)
            if matched:
                labels.append(rule.label)
                values |= rule.conditions
        if labels:
            conditions.append(Condition("weather", "·".join(labels), weather=frozenset(values)))
        return text

    @staticmethod
    def _add_ranges(text: str, rules: Sequence, kind: str, conditions: list[Condition]) -> str:
        """기온·기분: 여러 규칙이 걸려도 맨 앞 하나만 조건으로 쓴다 (서로 어긋나는 말이라서)."""
        chosen = None
        for rule in rules:
            matched, text = _consume(text, rule.patterns)
            if matched and chosen is None:
                chosen = rule
        if chosen is not None:
            conditions.append(Condition(kind, chosen.label, low=chosen.low, high=chosen.high))
        return text

    def _add_song_moods(self, text: str, conditions: list[Condition]) -> None:
        tags: list[str] = []
        for rule in self._rules.song_moods:
            matched, text = _consume(text, rule.patterns)
            if matched:
                tags.append(rule.tag)
        if tags:
            conditions.append(Condition("song_mood", f"{'·'.join(tags)} 곡", names=tuple(tags)))


def _consume(text: str, patterns: Sequence[re.Pattern[str]]) -> tuple[bool, str]:
    matched = False
    for pattern in patterns:
        while (match := pattern.search(text)) is not None:
            text = blank_out(text, match.start(), match.end())
            matched = True
    return matched, text


def _find_dongs(text: str) -> tuple[list[str], str]:
    found: list[str] = []
    while (match := _DONG.search(text)) is not None:
        if match.group(1) not in found:
            found.append(match.group(1))
        text = blank_out(text, match.start(), match.end())
    return found, text
