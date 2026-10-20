"""질문에서 뽑은 조건 — 기능5 기억 검색. 해석(question_parser)이 만들고 점수 계산이 쓴다."""

from dataclasses import dataclass
from datetime import date
from typing import Literal

from app.components.place_dictionary import PlaceEntry

ConditionKind = Literal[
    "place",  # 명소 사전의 장소 → 좌표 거리 (근처 기록이 하나도 없으면 코멘트 글자로)
    "place_text",  # 사전에 없는 장소 말("광교산") → 코멘트에 그 글자가 있는지
    "region",  # 시도·시군구 → 법정동코드 앞부분
    "dong",  # "XX동" → 법정동 이름
    "period",  # 지난달·작년 가을처럼 날짜 범위 → 기록한 날짜
    "season",  # 계절·N월 → 기록한 달 (해는 따지지 않는다)
    "day",  # 해 없이 "10월 15일" → 기록한 월·일 (해는 따지지 않는다)
    "weather",  # 날씨 → 기록 당시 날씨
    "time",  # 아침·저녁·퇴근길 같은 시간대 → 기록한 시각(시)
    "weekday",  # 주말·평일·금요일 → 기록한 요일
    "temperature",  # 기온 표현 → 기록 당시 기온
    "mood",  # 사용자가 입력한 기분 점수
    "song_mood",  # 곡 무드 태그 (잔잔한, 신나는 …)
    "artist",  # 이 사용자가 저장한 곡의 가수 이름 → 곡의 가수
    "title",  # 이 사용자가 저장한 곡의 제목 → 곡의 제목
    "genre",  # 장르 말 (발라드, 힙합 …) → 곡 장르 태그
]

# 장소·지역·동은 "어디서"라는 같은 종류의 조건이라 점수에서 같은 가중치를 받는다
LOCATION_KINDS: frozenset[str] = frozenset({"place", "place_text", "region", "dong"})


@dataclass(frozen=True)
class Condition:
    kind: ConditionKind
    label: str  # 안내 문장에 쓰는 이름 ("가을", "서울숲 근처")
    places: tuple[PlaceEntry, ...] = ()  # place
    prefixes: tuple[str, ...] = ()  # region: 법정동코드 앞부분 후보
    # dong: 동 이름 / song_mood: 곡 무드 태그 / place_text: 사전에 없는 장소 말 /
    # artist·title: 질문에서 찾은 가수·곡명 / genre: 장르 태그 /
    # place: 근처 기록이 없는 사전 장소의 이름·별칭 (resolve_conditions가 채운다)
    names: tuple[str, ...] = ()
    ranges: tuple[tuple[date, date], ...] = ()  # period: (시작일, 종료일) 둘 다 포함
    months: frozenset[int] = frozenset()  # season
    month_days: frozenset[tuple[int, int]] = frozenset()  # day: (월, 일)
    weather: frozenset[str] = frozenset()  # weather: records.weather_condition 값
    hours: frozenset[int] = frozenset()  # time: 맞는 시각(0~23시). 자정을 넘기는 구간도 풀어 둔다
    weekdays: frozenset[int] = frozenset()  # weekday: 월=0 … 일=6 (date.weekday())
    low: float | None = None  # temperature·mood 범위 (양끝 포함)
    high: float | None = None
    # region 전용. 약칭("수원", "공주")처럼 일반 낱말과 겹칠 수 있는 이름이면 False
    reliable: bool = True
