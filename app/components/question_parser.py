"""질문 해석 — 기능5 기억 검색. 질문 글자에서 "무엇을 찾는 질문인지"를 조건 목록으로 뽑는다.

DB를 부르지 않고 질문 글자와 우리 서버의 규칙 파일(places.yaml, regions.yaml,
question_rules.yaml)만 본다. 자물쇠에 저장된 값과 맞는지는 뒤 단계(condition_scoring)가 비교한다.

한 번 잡힌 글자는 공백으로 지워서 다른 규칙이 다시 쓰지 못하게 한다. 그래서 순서가 중요하다:
기간 → 장소 → 지역 → 동 → 계절·달 → 날씨 → 요일 → 시간대 → 기온 → 기분 → 장르 → 곡 무드 →
(남은 글자에서) 이 사용자의 가수·곡명 → (남은 글자에서) 사전에 없는 장소.
(예: "서울숲"을 장소로 가져가야 "서울"이 지역으로 또 잡히지 않고, "우울했을 때"를 기분으로
가져가야 "우울"이 곡 무드로 또 잡히지 않는다.)
"""

import calendar
import re
from collections.abc import Sequence
from datetime import date

from app.components.name_matcher import FOLLOWING, NOT_AFTER_HANGUL, blank_out
from app.components.period_parser import parse_periods
from app.components.place_dictionary import PlaceDictionary
from app.components.query_conditions import Condition
from app.components.question_rules import QuestionRules
from app.components.region_table import RegionTable
from app.components.track_lexicon import NameMatch, TrackLexicon
from app.components.unknown_places import find_unknown_places

# "연남동", "성수동1가", "종로3가" 모양. 실제 동인지는 여기서 모른다 — "운동을", "감동이"도 잡히고,
# 이 사용자의 기록에 있는 동인지는 뒤 단계(resolve_conditions)가 확인해서 걸러낸다.
_DONG = re.compile(NOT_AFTER_HANGUL + r"([가-힣]{1,6}동(?:\d가)?|[가-힣]{1,6}\d가)" + FOLLOWING)
_MONTH = re.compile(r"(?<![0-9])(\d{1,2})\s*월")
# 해 없이 "10월 15일" (해까지 말한 "2025년 10월 15일"·"작년 10월 15일"은 기간에서 이미 처리됨)
_MONTH_DAY = re.compile(r"(?<![0-9])(\d{1,2})\s*월\s*(\d{1,2})\s*일")
# 이름 바로 뒤에 붙어 "언제"를 말하는 꼴 ("가을 아침에", "소나기 올 때")
_TEMPORAL_AFTER = re.compile(r"\s*(?:에는|에도|에|엔|때|무렵|동안|부터|까지)")


class QuestionParser:
    def __init__(self, places: PlaceDictionary, regions: RegionTable, rules: QuestionRules):
        self._places = places
        self._regions = regions
        self._rules = rules

    @classmethod
    def load_default(cls) -> "QuestionParser":
        return cls(PlaceDictionary.load(), RegionTable.load(), QuestionRules.load())

    def parse(self, text: str, today: date, lexicon: TrackLexicon | None = None) -> list[Condition]:
        """질문에서 조건들을 뽑는다. 조건이 없으면 빈 목록. today는 "지난달"의 기준일.

        lexicon은 이 사용자가 저장한 곡의 가수·곡명 사전이다. 없으면 가수·곡명은 찾지 않는다.
        """
        artists: list[str] = []
        titles: list[str] = []
        if lexicon:
            artists, titles, text = self._claim_marked_names(text, today, lexicon)
        conditions, text = self._rule_conditions(text, today)
        if lexicon:
            text = self._add_track_names(text, lexicon, conditions, artists, titles)

        # 다 가져간 뒤에도 장소 같은 말이 남았으면 사전에 없는 장소다 — 코멘트에서 찾는다
        unknown = find_unknown_places(text)
        if unknown:
            conditions.append(Condition("place_text", "·".join(unknown), names=tuple(unknown)))
        return conditions

    def _rule_conditions(self, text: str, today: date) -> tuple[list[Condition], str]:
        """규칙 파일과 사전으로 뽑는 조건들 (기간 … 곡 무드). 가져간 글자를 지운 질문도 돌려준다."""
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

        text = self._add_month_days(text, conditions)
        text = self._add_seasons(text, conditions)
        text = self._add_weather(text, conditions)
        text = self._add_weekdays(text, conditions)
        text = self._add_times(text, conditions)
        text = self._add_ranges(text, self._rules.temperature, "temperature", conditions)
        text = self._add_ranges(text, self._rules.mood, "mood", conditions)
        text = self._add_genres(text, conditions)
        text = self._add_song_moods(text, conditions)
        return conditions, text

    def _claim_marked_names(
        self, text: str, today: date, lexicon: TrackLexicon
    ) -> tuple[list[str], list[str], str]:
        """곡명·가수임이 분명한 이름을 규칙보다 먼저 가져간다. (가수들, 곡명들, 지운 질문).

        규칙 단어(가을·소나기·성수동 …)와 겹치는 이름만 문제다. 규칙을 먼저 두는 게 기본이고
        ("가을에 들은 노래"의 가을은 계절이다), 아래 경우에만 이름이 이긴다.
        - 따옴표로 감쌌거나 "X라는 노래"처럼 이름이라고 분명히 말했다 ("가을이라는 노래")
        - 이름이 규칙 단어보다 의미 있게 길다 ("가을 아침" 속 "가을"은 계절이 아니다. 규칙이 가져간
          뒤 두 글자 이상의 낱말이 남아야 한다). 단, 뒤에 "에·때·무렵"이 붙으면 "언제"를 말한
          것이라 규칙이 이긴다 ("가을 아침에 들은 노래")
        - 규칙 단어처럼 보이는 것이 동 이름 후보뿐이다 ("운동"). 동 후보는 나중에 이 사용자의
          기록에 있는지 확인되어 걸러지므로, 곡명이 먼저 가져가는 편이 낫다.
        """
        artists: list[str] = []
        titles: list[str] = []
        for match in sorted(lexicon.matches(text), key=lambda m: -(m.end - m.start)):
            if not _still_there(text, match):
                continue  # 더 앞서 가져간 이름과 겹친다
            if not (match.explicit or self._name_beats_rules(text, match, today)):
                continue
            text = blank_out(text, match.start, match.end)
            bucket = artists if match.kind == "artist" else titles
            if match.name not in bucket:
                bucket.append(match.name)
        return artists, titles, text

    def _name_beats_rules(self, text: str, match: NameMatch, today: date) -> bool:
        conditions, rest = self._rule_conditions(text[match.start : match.end], today)
        kinds = {c.kind for c in conditions}
        if not kinds:
            return False  # 규칙과 안 겹친다 — 규칙이 끝난 뒤 남은 글자에서 평소처럼 찾는다
        if kinds == {"dong"}:
            return True
        if not any(len(word) >= 2 for word in rest.split()):
            # 이름 전체가 규칙 단어이거나 조사 같은 한 글자만 남는다
            # ("가을", "소나기", "비 오는 날", "봄날") — 규칙이 이긴다
            return False
        return _TEMPORAL_AFTER.match(text, match.end) is None

    @staticmethod
    def _add_month_days(text: str, conditions: list[Condition]) -> str:
        """해 없이 "10월 15일"처럼 월·일을 말한 경우.

        달만 보는 계절·달 규칙이 "15일"을 버리기 전에 먼저 가져간다.

        해는 따지지 않는다 (모든 해의 10월 15일). 없는 날짜("2월 30일")는 건드리지 않아서
        뒤의 달 규칙이 "2월"만 쓴다.
        """
        found: list[tuple[int, int]] = []

        def take(match: re.Match[str]) -> str:
            month, day = int(match.group(1)), int(match.group(2))
            if not (1 <= month <= 12 and 1 <= day <= calendar.monthrange(2000, month)[1]):
                return match.group(0)  # 윤년 2000년으로 확인해서 2월 29일은 인정한다
            if (month, day) not in found:
                found.append((month, day))
            return " " * len(match.group(0))

        text = _MONTH_DAY.sub(take, text)
        if found:
            label = "·".join(f"{m}월 {d}일" for m, d in found)
            conditions.append(Condition("day", label, month_days=frozenset(found)))
        return text

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

    def _add_weekdays(self, text: str, conditions: list[Condition]) -> str:
        labels: list[str] = []
        days: set[int] = set()
        for rule in self._rules.weekdays:
            matched, text = _consume(text, rule.patterns)
            if matched:
                labels.append(rule.label)
                days |= rule.days
        if labels:
            conditions.append(Condition("weekday", "·".join(labels), weekdays=frozenset(days)))
        return text

    def _add_times(self, text: str, conditions: list[Condition]) -> str:
        """시간대: 여러 개를 말하면("아침이나 저녁") 그중 하나에 맞으면 된다."""
        labels: list[str] = []
        hours: set[int] = set()
        for rule in self._rules.times:
            matched, text = _consume(text, rule.patterns)
            if matched:
                labels.append(rule.label)
                hours |= rule.hours
        if labels:
            conditions.append(Condition("time", "·".join(labels), hours=frozenset(hours)))
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

    def _add_genres(self, text: str, conditions: list[Condition]) -> str:
        tags: list[str] = []
        for rule in self._rules.genres:
            matched, text = _consume(text, rule.patterns)
            if matched:
                tags.append(rule.tag)
        if tags:
            conditions.append(Condition("genre", f"{'·'.join(tags)} 장르", names=tuple(tags)))
        return text

    @staticmethod
    def _add_track_names(
        text: str,
        lexicon: TrackLexicon,
        conditions: list[Condition],
        artists: list[str],
        titles: list[str],
    ) -> str:
        """남은 글자에서 이 사용자가 저장한 곡의 가수·곡명을 더 찾는다 (가수가 곡명보다 먼저).

        artists·titles는 규칙보다 먼저 가져간 이름들이다. 같은 종류끼리 한 조건으로 합친다.
        """
        found, text = lexicon.find_artists(text)
        artists = [*artists, *(n for n in found if n not in artists)]
        if artists:
            label = f"가수 {'·'.join(artists)}"
            conditions.append(Condition("artist", label, names=tuple(artists)))
        found, text = lexicon.find_titles(text)
        titles = [*titles, *(n for n in found if n not in titles)]
        if titles:
            label = f"곡 {'·'.join(titles)}"
            conditions.append(Condition("title", label, names=tuple(titles)))
        return text

    def _add_song_moods(self, text: str, conditions: list[Condition]) -> str:
        tags: list[str] = []
        for rule in self._rules.song_moods:
            matched, text = _consume(text, rule.patterns)
            if matched:
                tags.append(rule.tag)
        if tags:
            conditions.append(Condition("song_mood", f"{'·'.join(tags)} 곡", names=tuple(tags)))
        return text


def _still_there(text: str, match: NameMatch) -> bool:
    """앞서 가져가서 지워진 글자가 아직 이 이름 자리에 남아 있는지."""
    return text[match.start : match.end].strip() != "" and "  " not in text[match.start : match.end]


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
