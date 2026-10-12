"""질문에서 뽑은 조건 — 기능5 기억 검색. 해석(question_parser)이 만들고 점수 계산이 쓴다."""

from dataclasses import dataclass
from datetime import date
from typing import Literal

from app.components.place_dictionary import PlaceEntry

ConditionKind = Literal[
    "place",  # 명소 사전의 장소 → 좌표 거리
    "region",  # 시도·시군구 → 법정동코드 앞부분
    "dong",  # "XX동" → 법정동 이름
    "period",  # 지난달·작년 가을처럼 날짜 범위 → 기록한 날짜
    "season",  # 계절·N월 → 기록한 달 (해는 따지지 않는다)
    "weather",  # 날씨 → 기록 당시 날씨
    "temperature",  # 기온 표현 → 기록 당시 기온
    "mood",  # 사용자가 입력한 기분 점수
    "song_mood",  # 곡 무드 태그 (잔잔한, 신나는 …)
]

# 장소·지역·동은 "어디서"라는 같은 종류의 조건이라 점수에서 같은 가중치를 받는다
LOCATION_KINDS: frozenset[str] = frozenset({"place", "region", "dong"})


@dataclass(frozen=True)
class Condition:
    kind: ConditionKind
    label: str  # 안내 문장에 쓰는 이름 ("가을", "서울숲 근처")
    places: tuple[PlaceEntry, ...] = ()  # place
    prefixes: tuple[str, ...] = ()  # region: 법정동코드 앞부분 후보
    names: tuple[str, ...] = ()  # dong: 동 이름 / song_mood: 곡 무드 태그
    ranges: tuple[tuple[date, date], ...] = ()  # period: (시작일, 종료일) 둘 다 포함
    months: frozenset[int] = frozenset()  # season
    weather: frozenset[str] = frozenset()  # weather: records.weather_condition 값
    low: float | None = None  # temperature·mood 범위 (양끝 포함)
    high: float | None = None
    # region 전용. 약칭("수원", "공주")처럼 일반 낱말과 겹칠 수 있는 이름이면 False
    reliable: bool = True
