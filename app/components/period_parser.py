"""기간 표현("지난달", "작년 가을")을 날짜 범위로 바꾼다 — 기능5 기억 검색 (순수 함수).

"오늘"이 무슨 날인지는 호출하는 쪽이 넘긴다(테스트에서 날짜를 고정하려고).
정한 약속:
- 한 주는 월요일~일요일이다.
- 겨울은 11월~다음 해 2월이라서 "작년 겨울"은 작년 11월 1일~올해 2월 말이다.
- "최근"·"요즘"은 오늘까지 30일이다.
- "지난 가을"은 이미 끝난 가장 가까운 가을, "이번 가을"은 올해 가을이다.
연도 없이 계절·달만 말하면 여기서 다루지 않는다(모든 해의 그 달로 비교한다).
"""

import calendar
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from datetime import date, timedelta

from app.components.name_matcher import blank_out

RECENT_DAYS = 30

_YEAR_OFFSETS = {"재작년": -2, "작년": -1, "지난해": -1, "올해": 0, "금년": 0, "내년": 1}
_NUMBER_WORDS = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6}
_BEFORE = r"(?<![가-힣0-9])"
_YEAR_WORDS = "|".join(sorted(_YEAR_OFFSETS, key=len, reverse=True))  # 긴 말부터 ("재작년"이 먼저)


# "지난 주말"에서 앞 낱말만 가져갔을 때의 이름 (안내 문장에 "지난, 주말"로 나오지 않게)
_WEEKEND_LEAD_LABELS = {"지난": "지난 주", "저번": "지난 주", "이번": "이번 주"}


@dataclass(frozen=True)
class PeriodMatch:
    label: str  # 질문에서 잡힌 말
    start: date
    end: date  # 이 날 포함


def parse_periods(
    text: str, today: date, seasons: Mapping[str, Collection[int]]
) -> tuple[list[PeriodMatch], str]:
    """기간 표현을 모두 찾는다. (찾은 기간들, 찾은 글자를 공백으로 지운 질문)을 돌려준다.

    seasons는 계절 이름 → 달 목록 ({"겨울": [11, 12, 1, 2]}). 연도와 함께 말한 계절에 쓴다.
    """
    season_names = "|".join(seasons)
    rules = _rules(today, seasons, season_names)
    found: list[PeriodMatch] = []
    for pattern, build in rules:
        while True:
            match = pattern.search(text)
            if match is None:
                break
            span = build(match)
            if span is not None:
                label = " ".join(match.group(0).split())
                label = _WEEKEND_LEAD_LABELS.get(label, label)
                found.append(PeriodMatch(label, span[0], span[1]))
            text = blank_out(text, match.start(), match.end())
    return found, text


def month_range(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


def week_range(day: date) -> tuple[date, date]:
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=6)


def season_range(months: Collection[int], year: int) -> tuple[date, date]:
    """year에 시작하는 계절의 날짜 범위. 겨울(11~2월)처럼 해를 넘기면 다음 해 말까지."""
    ordered = sorted(months)
    if 12 in ordered and 1 in ordered:  # 해를 넘기는 계절
        start_month = min(m for m in ordered if m >= 7)
        end_month = max(m for m in ordered if m <= 6)
        return date(year, start_month, 1), month_range(year + 1, end_month)[1]
    return date(year, ordered[0], 1), month_range(year, ordered[-1])[1]


def _shift_month(year: int, month: int, delta: int) -> tuple[int, int]:
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


Span = tuple[date, date] | None


def _rules(
    today: date, seasons: Mapping[str, Collection[int]], season_names: str
) -> list[tuple[re.Pattern[str], Callable[[re.Match[str]], Span]]]:
    def full_date(m: re.Match[str]) -> Span:
        day = _safe_date(int(m[1]), int(m[2]), int(m[3]))
        return (day, day) if day else None

    def year_month(m: re.Match[str]) -> Span:
        return month_range(int(m[1]), int(m[2])) if 1 <= int(m[2]) <= 12 else None

    def year_only(m: re.Match[str]) -> Span:
        return date(int(m[1]), 1, 1), date(int(m[1]), 12, 31)

    def year_word_season(m: re.Match[str]) -> Span:
        return season_range(seasons[m[2]], today.year + _YEAR_OFFSETS[m[1]])

    def year_word_month(m: re.Match[str]) -> Span:
        month = int(m[2])
        return month_range(today.year + _YEAR_OFFSETS[m[1]], month) if 1 <= month <= 12 else None

    def relative_season(m: re.Match[str]) -> Span:
        months = seasons[m[2]]
        if m[1] in ("지난", "저번"):  # 이미 끝난 가장 가까운 계절
            ranges = [season_range(months, today.year - back) for back in (0, 1, 2)]
            finished = [r for r in ranges if r[1] < today]
            return max(finished, key=lambda r: r[1])
        wraps = 12 in months and 1 in months
        year = today.year - 1 if wraps and today.month <= 2 else today.year
        return season_range(months, year)

    def year_word(m: re.Match[str]) -> Span:
        year = today.year + _YEAR_OFFSETS[m[1]]
        return date(year, 1, 1), date(year, 12, 31)

    def last_month(_: re.Match[str]) -> Span:
        return month_range(*_shift_month(today.year, today.month, -1))

    def this_month(_: re.Match[str]) -> Span:
        return month_range(today.year, today.month)

    def last_week(_: re.Match[str]) -> Span:
        return week_range(today - timedelta(days=7))

    def this_week(_: re.Match[str]) -> Span:
        return week_range(today)

    def day_offset(days: int) -> Callable[[re.Match[str]], Span]:
        def build(_: re.Match[str]) -> Span:
            day = today + timedelta(days=days)
            return day, day

        return build

    def n_ago(m: re.Match[str]) -> Span:
        n = int(m[1]) if m[1].isdigit() else _NUMBER_WORDS[m[1]]
        unit = m[2]
        if unit == "일":
            day = today - timedelta(days=n)
            return day, day
        if unit == "주":
            return week_range(today - timedelta(days=7 * n))
        if unit == "년":
            return date(today.year - n, 1, 1), date(today.year - n, 12, 31)
        return month_range(*_shift_month(today.year, today.month, -n))  # 개월·달

    def recent(_: re.Match[str]) -> Span:
        return today - timedelta(days=RECENT_DAYS - 1), today

    numbers = "|".join(["\\d+", *_NUMBER_WORDS])
    return [
        (re.compile(r"(?<![0-9])(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일"), full_date),
        (re.compile(r"(?<![0-9])(\d{4})\s*년\s*(\d{1,2})\s*월"), year_month),
        (re.compile(r"(?<![0-9])((?:19|20)\d{2})\s*년"), year_only),
        (re.compile(_BEFORE + rf"({_YEAR_WORDS})\s*({season_names})"), year_word_season),
        (re.compile(_BEFORE + rf"({_YEAR_WORDS})\s*(\d{{1,2}})\s*월"), year_word_month),
        (re.compile(_BEFORE + rf"(지난|저번|이번|올)\s*({season_names})"), relative_season),
        (re.compile(_BEFORE + rf"({_YEAR_WORDS})"), year_word),
        (re.compile(_BEFORE + r"(?:지난\s*달|저번\s*달)"), last_month),
        (re.compile(_BEFORE + r"(?:이번\s*달|이달)"), this_month),
        # "지난 주말": 앞 낱말만 가져가서 그 주로 쓰고, "주말"은 남겨 요일 규칙(weekday)이 잡게 한다
        (re.compile(_BEFORE + r"(?:지난|저번)\s*(?=주\s*말)"), last_week),
        (re.compile(_BEFORE + r"이번\s*(?=주\s*말)"), this_week),
        (re.compile(_BEFORE + r"(?:지난\s*주|저번\s*주)(?!말)"), last_week),
        (re.compile(_BEFORE + r"이번\s*주(?!말)"), this_week),
        (re.compile(_BEFORE + rf"({numbers})\s*(일|주|개월|달|년)\s*전"), n_ago),
        (re.compile(_BEFORE + r"(?:그저께|그제)"), day_offset(-2)),
        (re.compile(_BEFORE + r"어제"), day_offset(-1)),
        (re.compile(_BEFORE + r"오늘"), day_offset(0)),
        (re.compile(_BEFORE + r"(?:최근|요즘|근래)"), recent),
    ]
