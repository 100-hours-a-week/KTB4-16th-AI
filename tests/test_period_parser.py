from datetime import date

import pytest

from app.components.period_parser import month_range, parse_periods, season_range, week_range

SEASONS = {"봄": [3, 4, 5], "여름": [6, 7, 8], "가을": [9, 10], "겨울": [11, 12, 1, 2]}
TODAY = date(2026, 10, 12)  # 월요일


def _parse(text: str, today: date = TODAY):
    periods, remaining = parse_periods(text, today, SEASONS)
    return [(p.label, p.start, p.end) for p in periods], remaining


def test_reference_day_is_a_monday():
    assert TODAY.weekday() == 0


@pytest.mark.parametrize(
    ("text", "label", "start", "end"),
    [
        ("지난달에 들은 노래", "지난달", date(2026, 9, 1), date(2026, 9, 30)),
        ("저번 달에 들은 노래", "저번 달", date(2026, 9, 1), date(2026, 9, 30)),
        ("이번 달에 들은 노래", "이번 달", date(2026, 10, 1), date(2026, 10, 31)),
        ("지난주에 들은 노래", "지난주", date(2026, 10, 5), date(2026, 10, 11)),  # 월~일
        ("저번 주에 들은 노래", "저번 주", date(2026, 10, 5), date(2026, 10, 11)),
        ("이번 주에 들은 노래", "이번 주", date(2026, 10, 12), date(2026, 10, 18)),
        ("어제 들은 노래", "어제", date(2026, 10, 11), date(2026, 10, 11)),
        ("그저께 들은 노래", "그저께", date(2026, 10, 10), date(2026, 10, 10)),
        ("오늘 들은 노래", "오늘", date(2026, 10, 12), date(2026, 10, 12)),
        ("작년에 들은 노래", "작년", date(2025, 1, 1), date(2025, 12, 31)),
        ("재작년에 들은 노래", "재작년", date(2024, 1, 1), date(2024, 12, 31)),
        ("지난해에 들은 노래", "지난해", date(2025, 1, 1), date(2025, 12, 31)),
        ("올해 들은 노래", "올해", date(2026, 1, 1), date(2026, 12, 31)),
        ("최근에 들은 노래", "최근", date(2026, 9, 13), date(2026, 10, 12)),  # 오늘까지 30일
    ],
)
def test_relative_expressions(text, label, start, end):
    found, _ = _parse(text)

    assert found == [(label, start, end)]


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("작년 가을에 들은 노래", date(2025, 9, 1), date(2025, 10, 31)),
        ("올해 여름에 들은 노래", date(2026, 6, 1), date(2026, 8, 31)),
        ("작년 겨울에 들은 노래", date(2025, 11, 1), date(2026, 2, 28)),  # 겨울은 해를 넘긴다
        ("재작년 겨울에 들은 노래", date(2024, 11, 1), date(2025, 2, 28)),
        ("내년 봄에 들은 노래", date(2027, 3, 1), date(2027, 5, 31)),
        ("작년 10월에 들은 노래", date(2025, 10, 1), date(2025, 10, 31)),
        ("지난 가을에 들은 노래", date(2025, 9, 1), date(2025, 10, 31)),  # 올가을은 아직 안 끝났다
        (
            "지난 여름에 들은 노래",
            date(2026, 6, 1),
            date(2026, 8, 31),
        ),  # 이미 끝난 가장 가까운 여름
        ("지난 겨울에 들은 노래", date(2025, 11, 1), date(2026, 2, 28)),
        ("이번 가을에 들은 노래", date(2026, 9, 1), date(2026, 10, 31)),
        ("올 겨울에 들은 노래", date(2026, 11, 1), date(2027, 2, 28)),
    ],
)
def test_year_with_season_or_month(text, start, end):
    found, _ = _parse(text)

    assert [(s, e) for _, s, e in found] == [(start, end)]


def test_this_winter_in_january_belongs_to_the_winter_that_started_last_year():
    found, _ = _parse("올 겨울에 들은 노래", today=date(2027, 1, 20))

    assert [(s, e) for _, s, e in found] == [(date(2026, 11, 1), date(2027, 2, 28))]


@pytest.mark.parametrize(
    ("text", "start", "end"),
    [
        ("2025년 10월 3일에 들은 노래", date(2025, 10, 3), date(2025, 10, 3)),
        ("2025년 10월에 들은 노래", date(2025, 10, 1), date(2025, 10, 31)),
        ("2025년에 들은 노래", date(2025, 1, 1), date(2025, 12, 31)),
        ("3일 전에 들은 노래", date(2026, 10, 9), date(2026, 10, 9)),
        ("2주 전에 들은 노래", date(2026, 9, 28), date(2026, 10, 4)),
        ("한 달 전에 들은 노래", date(2026, 9, 1), date(2026, 9, 30)),
        ("3개월 전에 들은 노래", date(2026, 7, 1), date(2026, 7, 31)),
        ("두 달 전에 들은 노래", date(2026, 8, 1), date(2026, 8, 31)),
        ("2년 전에 들은 노래", date(2024, 1, 1), date(2024, 12, 31)),
    ],
)
def test_explicit_dates_and_n_units_ago(text, start, end):
    found, _ = _parse(text)

    assert [(s, e) for _, s, e in found] == [(start, end)]


def test_month_rollover_across_new_year():
    today = date(2026, 1, 15)

    assert [(s, e) for _, s, e in _parse("지난달", today)[0]] == [
        (date(2025, 12, 1), date(2025, 12, 31))
    ]
    assert [(s, e) for _, s, e in _parse("3개월 전", today)[0]] == [
        (date(2025, 10, 1), date(2025, 10, 31))
    ]


def test_leap_february():
    found, _ = _parse("지난달", today=date(2024, 3, 10))

    assert [(s, e) for _, s, e in found] == [(date(2024, 2, 1), date(2024, 2, 29))]


def test_several_periods_in_one_question():
    found, _ = _parse("작년이랑 올해 여름에 들은 노래")

    assert sorted(label for label, _, _ in found) == ["올해 여름", "작년"]


@pytest.mark.parametrize(
    "text",
    [
        "비 오던 날 들은 노래",
        "메시지를 전달받은 날 들은 노래",  # "전달"은 지난달이 아니다
        "지난 주말에 들은 노래",  # 주말은 이번 단계에서 다루지 않는다
        "가을에 들은 노래",  # 연도 없는 계절은 계절 규칙이 처리한다
        "서울숲에서 들은 노래",
    ],
)
def test_no_period_in_question(text):
    found, remaining = _parse(text)

    assert found == []
    assert remaining == text


def test_matched_text_is_blanked_out():
    _, remaining = _parse("작년 가을에 들은 노래")

    assert "작년" not in remaining and "가을" not in remaining
    assert len(remaining) == len("작년 가을에 들은 노래")


def test_helpers():
    assert month_range(2026, 2) == (date(2026, 2, 1), date(2026, 2, 28))
    assert week_range(date(2026, 10, 14)) == (date(2026, 10, 12), date(2026, 10, 18))
    assert season_range([3, 4, 5], 2026) == (date(2026, 3, 1), date(2026, 5, 31))
    assert season_range([11, 12, 1, 2], 2026) == (date(2026, 11, 1), date(2027, 2, 28))
