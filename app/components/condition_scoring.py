"""조건 점수 — 기능5 기억 검색 (순수 함수, DB·외부 호출 없음).

질문에서 뽑은 조건(question_parser)이 자물쇠에 저장된 값과 맞는지 비교해서, 자물쇠마다
"조건 중 무엇이 맞았는지"와 0~1 점수를 낸다. 조건 하나의 점수는 대부분 맞으면 1, 아니면 0이고
장소만 중심에 가까울수록 높다(거리 ÷ 반경). 좌표로 비교할 수 없는 장소(사전에 없는 말, 또는
근처 기록이 없는 사전 장소)는 코멘트에 그 이름이 있으면 맞는 것으로 본다.

질문에 나온 조건이 모두 맞는 기록과 일부만 맞는 기록을 나누는 것은 여기서 정한 `matched`·
`missed`를 보고 순위를 합치는 쪽(rank_fusion)이 한다.
"""

import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from app.components.geo import haversine_km, proximity_score
from app.components.name_matcher import name_pattern, substring_pattern
from app.components.place_dictionary import PlaceEntry
from app.components.query_conditions import LOCATION_KINDS, Condition
from app.components.track_lexicon import artist_keys, compact, title_keys
from app.db.repositories.memory_record_repository import ConditionRow


@dataclass(frozen=True)
class ConditionWeights:
    """조건 종류별 가중치. 장소·지역·동("어디서")이 나머지보다 크다."""

    location: float = 2.0
    other: float = 1.0


@dataclass(frozen=True)
class RecordScore:
    record_id: int
    score: float  # 0~1, 질문에 나온 조건들의 가중 평균
    matched: tuple[str, ...]  # 맞은 조건의 이름
    missed: tuple[str, ...]  # 안 맞은 조건의 이름

    @property
    def all_matched(self) -> bool:
        return bool(self.matched) and not self.missed


def resolve_conditions(
    conditions: Sequence[Condition], rows: Sequence[ConditionRow]
) -> list[Condition]:
    """질문에서 뽑은 지역·동 후보를 이 사용자의 자물쇠로 확인해서 정리한다.

    - 동: "운동을", "감동이"처럼 동 이름 모양일 뿐인 말은 이 사용자의 기록에 없어서 버린다.
    - 지역: 같은 이름이 여러 곳에 있으면("중구") 이 사용자의 기록에 있는 곳만 남긴다.
      기록이 하나도 없는 지역은, 공식 이름이면 그대로 둔다 — "성동구 기록이 없다"가 답이므로.
      '시'를 뗀 약칭("공주")은 일반 낱말일 수 있어서 기록이 없으면 버린다.
    - 장소(사전): 반경 안에 이 사용자의 기록이 하나도 없는 장소는 좌표로 비교할 수가 없다.
      그런 장소는 이름·별칭을 코멘트에서 찾도록 names에 적어 둔다 ("성수역에서 저장했지만
      코멘트에 '서울숲'이라고 쓴 기록"). 반경 안에 기록이 있는 장소는 좌표만 쓴다.
    - 장소(사전 밖): 이 사용자의 코멘트 어디에도 안 나오는 말은 버린다. 찾을 방법이 없는 조건을
      그대로 두면 모든 기록이 "조건에 안 맞음"이 되어 다른 조건으로 찾은 결과까지 사라진다.
      (버린 말은 호출하는 쪽이 안내 문장으로 알린다 — dropped_place_words)
    """
    resolved: list[Condition] = []
    for cond in conditions:
        if cond.kind == "place":
            resolved.append(_with_comment_fallback(cond, rows))
        elif cond.kind == "place_text":
            mentioned = tuple(w for w in cond.names if _mentioned_anywhere(w, rows))
            if mentioned:
                resolved.append(replace(cond, names=mentioned, label="·".join(mentioned)))
        elif cond.kind == "dong":
            names = tuple(
                n
                for n in cond.names
                if any(r.dong_name and r.dong_name.startswith(n) for r in rows)
            )
            if names:
                resolved.append(replace(cond, names=names, label="·".join(names)))
        elif cond.kind == "region":
            present = tuple(
                p
                for p in cond.prefixes
                if any(r.dong_code and r.dong_code.startswith(p) for r in rows)
            )
            if present:
                resolved.append(replace(cond, prefixes=present))
            elif cond.reliable:
                resolved.append(cond)
        else:
            resolved.append(cond)
    return resolved


def dropped_place_words(asked: Sequence[Condition], resolved: Sequence[Condition]) -> list[str]:
    """resolve_conditions가 버린 사전 밖 장소 말. asked는 resolve 전, resolved는 후의 조건."""
    kept = {w for c in resolved if c.kind == "place_text" for w in c.names}
    return [w for c in asked if c.kind == "place_text" for w in c.names if w not in kept]


def kept_place_words(resolved: Sequence[Condition]) -> list[str]:
    """코멘트에서 실제로 찾아 조건으로 쓰는 사전 밖 장소 말."""
    return [w for c in resolved if c.kind == "place_text" for w in c.names]


def score_rows(
    rows: Iterable[ConditionRow],
    conditions: Sequence[Condition],
    *,
    moods: Mapping[int, Sequence[str]],
    music_similarity: Mapping[int, float],
    min_similarity: float,
    weights: ConditionWeights,
    genres: Mapping[int, str] | None = None,
) -> dict[int, RecordScore]:
    """자물쇠마다 조건 점수를 낸다. 조건이 없으면 빈 결과.

    moods: 자물쇠마다 곡 무드 태그. music_similarity: 자물쇠마다 질문과 곡 분위기의 유사도 —
    곡에 무드 태그가 아직 없을 때 "잔잔한 곡" 같은 조건을 이 유사도로 대신 판단한다.
    genres: 자물쇠마다 곡 장르 태그 (장르 조건용).
    """
    if not conditions:
        return {}

    patterns = [_comment_patterns(cond) for cond in conditions]  # 조건마다 한 번만 만든다
    scores: dict[int, RecordScore] = {}
    for row in rows:
        parts = [
            (
                cond,
                _condition_score(
                    cond,
                    row,
                    tags=moods.get(row.record_id, ()),
                    genre=(genres or {}).get(row.record_id),
                    music_similarity=music_similarity.get(row.record_id),
                    min_similarity=min_similarity,
                    mentioned=_mentions(row.comment, pats),
                ),
            )
            for cond, pats in zip(conditions, patterns, strict=True)
        ]
        total = sum(_weight(cond, weights) for cond, _ in parts)
        score = sum(_weight(cond, weights) * s for cond, s in parts) / total
        scores[row.record_id] = RecordScore(
            record_id=row.record_id,
            score=score,
            matched=tuple(cond.label for cond, s in parts if s > 0),
            missed=tuple(cond.label for cond, s in parts if s <= 0),
        )
    return scores


def top_candidates(
    scores: Mapping[int, RecordScore], *, exclude: Collection[int], limit: int
) -> list[int]:
    """조건 점수가 0보다 큰 기록 중 exclude에 없는 것을 점수 높은 순으로 limit개.

    벡터 검색 상위 후보에 못 든 기록 중에서 따로 후보로 더할 것을 고른다.
    """
    wanted = [s for s in scores.values() if s.score > 0 and s.record_id not in exclude]
    wanted.sort(key=lambda s: (-s.score, -s.record_id))
    return [s.record_id for s in wanted[:limit]]


def _weight(cond: Condition, weights: ConditionWeights) -> float:
    return weights.location if cond.kind in LOCATION_KINDS else weights.other


def _condition_score(
    cond: Condition,
    row: ConditionRow,
    *,
    tags: Sequence[str],
    genre: str | None = None,
    music_similarity: float | None,
    min_similarity: float,
    mentioned: bool,
) -> float:
    """mentioned: 이 조건의 이름이 이 기록의 코멘트에 있는지 (place·place_text만 쓴다)."""
    match cond.kind:
        case "place":
            # 좌표가 가까우면 거리로, 아니면 코멘트에 이름이 있을 때 1
            # (코멘트로 찾는 이름은 근처 기록이 없는 장소만 names에 있다)
            return max(_nearest_place_score(cond, row), _flag(mentioned))
        case "place_text":
            return _flag(mentioned)
        case "region":
            return _flag(
                bool(row.dong_code) and any(row.dong_code.startswith(p) for p in cond.prefixes)
            )
        case "dong":
            return _flag(
                bool(row.dong_name) and any(row.dong_name.startswith(n) for n in cond.names)
            )
        case "period":
            day = row.created_at.date()
            return _flag(any(start <= day <= end for start, end in cond.ranges))
        case "season":
            return _flag(row.created_at.month in cond.months)
        case "day":
            return _flag((row.created_at.month, row.created_at.day) in cond.month_days)
        case "weather":
            return _flag(_weather_code(row.weather_condition) in cond.weather)
        case "time":
            return _flag(row.created_at.hour in cond.hours)
        case "weekday":
            return _flag(row.created_at.weekday() in cond.weekdays)
        case "temperature":
            return _flag(_in_range(row.temperature, cond))
        case "mood":
            return _flag(_in_range(row.mood_score, cond))
        case "artist":
            return _flag(bool(artist_keys(row.artist_name) & _name_keys(cond)))
        case "title":
            return _flag(bool(title_keys(row.track_title) & _name_keys(cond)))
        case "genre":
            return _flag(genre is not None and genre in cond.names)
        case "song_mood":
            if tags:  # 태그가 있으면 태그로 판단한다
                return _flag(any(tag in cond.names for tag in tags))
            # 태그가 아직 없는 곡: 곡 분위기 벡터가 질문과 충분히 비슷하면 맞는 것으로 본다
            return _flag(music_similarity is not None and music_similarity >= min_similarity)
    raise ValueError(f"알 수 없는 조건 종류: {cond.kind}")


def _place_score(place: PlaceEntry, row: ConditionRow) -> float:
    if row.latitude is None or row.longitude is None:
        return 0.0
    return proximity_score(
        haversine_km(place.lat, place.lng, row.latitude, row.longitude), place.radius_km
    )


def _nearest_place_score(cond: Condition, row: ConditionRow) -> float:
    return max(_place_score(place, row) for place in cond.places)


def _with_comment_fallback(cond: Condition, rows: Sequence[ConditionRow]) -> Condition:
    """반경 안에 이 사용자의 기록이 하나도 없는 장소의 이름·별칭을 코멘트 검색용으로 적어 둔다."""
    silent = [
        place for place in cond.places if not any(_place_score(place, row) > 0 for row in rows)
    ]
    if not silent:
        return cond
    return replace(cond, names=tuple(n for place in silent for n in (place.name, *place.aliases)))


def _comment_patterns(cond: Condition) -> list[re.Pattern[str]]:
    """코멘트에서 이름을 찾는 규칙. 사전 장소는 질문과 같은 앞뒤 규칙, 사전 밖 말은 포함 여부."""
    if cond.kind == "place":
        return [name_pattern(n) for n in cond.names]
    if cond.kind == "place_text":
        return [substring_pattern(n) for n in cond.names]
    return []


def _mentions(comment: str | None, patterns: Sequence[re.Pattern[str]]) -> bool:
    return bool(comment) and any(p.search(comment) for p in patterns)


def _mentioned_anywhere(word: str, rows: Sequence[ConditionRow]) -> bool:
    pattern = substring_pattern(word)
    return any(_mentions(row.comment, [pattern]) for row in rows)


def _name_keys(cond: Condition) -> frozenset[str]:
    """가수·곡명 조건의 이름을 비교용 글자로 (공백·대소문자·기호 무시)."""
    return frozenset(k for k in (compact(n) for n in cond.names) if k)


def _weather_code(value: str | None) -> str | None:
    """DB의 날씨 값을 비교용으로 정리한다 — 앞뒤 공백과 대소문자는 무시한다.

    규칙 파일의 값(CLOUDY 등)은 대문자라서, 저장된 값이 "cloudy"나 " CLOUDY"여도 같은 날씨로 본다.
    """
    return None if value is None else value.strip().upper()


def _flag(matched: bool) -> float:
    return 1.0 if matched else 0.0


def _in_range(value: float | None, cond: Condition) -> bool:
    return (
        value is not None
        and cond.low is not None
        and cond.high is not None
        and (cond.low <= value <= cond.high)
    )
