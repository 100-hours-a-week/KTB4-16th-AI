"""조건 점수 — 기능5 기억 검색 (순수 함수, DB·외부 호출 없음).

질문에서 뽑은 조건(question_parser)이 자물쇠에 저장된 값과 맞는지 비교해서, 자물쇠마다
"조건 중 무엇이 맞았는지"와 0~1 점수를 낸다. 조건 하나의 점수는 대부분 맞으면 1, 아니면 0이고
장소만 중심에 가까울수록 높다(거리 ÷ 반경).

질문에 나온 조건이 모두 맞는 기록과 일부만 맞는 기록을 나누는 것은 여기서 정한 `matched`·
`missed`를 보고 순위를 합치는 쪽(rank_fusion)이 한다.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace

from app.components.geo import haversine_km, proximity_score
from app.components.query_conditions import LOCATION_KINDS, Condition
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
    """
    resolved: list[Condition] = []
    for cond in conditions:
        if cond.kind == "dong":
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


def score_rows(
    rows: Iterable[ConditionRow],
    conditions: Sequence[Condition],
    *,
    moods: Mapping[int, Sequence[str]],
    music_similarity: Mapping[int, float],
    min_similarity: float,
    weights: ConditionWeights,
) -> dict[int, RecordScore]:
    """자물쇠마다 조건 점수를 낸다. 조건이 없으면 빈 결과.

    moods: 자물쇠마다 곡 무드 태그. music_similarity: 자물쇠마다 질문과 곡 분위기의 유사도 —
    곡에 무드 태그가 아직 없을 때 "잔잔한 곡" 같은 조건을 이 유사도로 대신 판단한다.
    """
    if not conditions:
        return {}

    scores: dict[int, RecordScore] = {}
    for row in rows:
        parts = [
            (
                cond,
                _condition_score(
                    cond,
                    row,
                    tags=moods.get(row.record_id, ()),
                    music_similarity=music_similarity.get(row.record_id),
                    min_similarity=min_similarity,
                ),
            )
            for cond in conditions
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
    music_similarity: float | None,
    min_similarity: float,
) -> float:
    match cond.kind:
        case "place":
            if row.latitude is None or row.longitude is None:
                return 0.0
            return max(
                proximity_score(
                    haversine_km(place.lat, place.lng, row.latitude, row.longitude),
                    place.radius_km,
                )
                for place in cond.places
            )
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
        case "weather":
            return _flag(row.weather_condition in cond.weather)
        case "temperature":
            return _flag(_in_range(row.temperature, cond))
        case "mood":
            return _flag(_in_range(row.mood_score, cond))
        case "song_mood":
            if tags:  # 태그가 있으면 태그로 판단한다
                return _flag(any(tag in cond.names for tag in tags))
            # 태그가 아직 없는 곡: 곡 분위기 벡터가 질문과 충분히 비슷하면 맞는 것으로 본다
            return _flag(music_similarity is not None and music_similarity >= min_similarity)
    raise ValueError(f"알 수 없는 조건 종류: {cond.kind}")


def _flag(matched: bool) -> float:
    return 1.0 if matched else 0.0


def _in_range(value: float | None, cond: Condition) -> bool:
    return (
        value is not None
        and cond.low is not None
        and cond.high is not None
        and (cond.low <= value <= cond.high)
    )
