import math

import pytest

from app.components.condition_scoring import RecordScore
from app.components.rank_fusion import FusedRecord, combine_with_conditions, fuse
from tests.memory_fakes import hit

DEFAULTS = {"comment_weight": 0.5, "music_weight": 0.5, "min_similarity": 0.4}


def test_record_found_by_both_searches_beats_one_found_by_only_one():
    fused = fuse([hit(1, 0.8), hit(2, 0.8)], [hit(1, 0.8)], **DEFAULTS)

    assert [f.record_id for f in fused] == [1, 2]
    assert math.isclose(fused[0].score, 0.8)  # (0.5*0.8 + 0.5*0.8) / 1
    assert math.isclose(fused[1].score, 0.4)  # 곡 쪽에서 안 잡히면 그 몫은 0
    assert fused[1].music_similarity is None


def test_score_is_weighted_average_so_stays_between_0_and_1():
    fused = fuse([hit(1, 0.9)], [hit(1, 0.5)], comment_weight=3, music_weight=1, min_similarity=0.4)

    assert math.isclose(fused[0].score, (3 * 0.9 + 1 * 0.5) / 4)
    assert 0 <= fused[0].score <= 1


def test_hit_below_min_similarity_is_dropped_from_that_search_only():
    # 1번: 코멘트 0.3(미달)·곡 0.9 → 곡 쪽만 남는다. 2번: 둘 다 미달 → 사라진다.
    fused = fuse([hit(1, 0.3), hit(2, 0.35)], [hit(1, 0.9), hit(2, 0.2)], **DEFAULTS)

    assert [f.record_id for f in fused] == [1]
    assert fused[0].comment_similarity is None
    assert fused[0].music_similarity == 0.9


def test_hit_exactly_at_min_similarity_is_kept():
    assert [f.record_id for f in fuse([hit(1, 0.4)], [], **DEFAULTS)] == [1]


def test_equal_scores_put_newer_record_first_so_order_is_stable():
    fused = fuse([hit(5, 0.7), hit(9, 0.7), hit(7, 0.7)], [], **DEFAULTS)

    assert [f.record_id for f in fused] == [9, 7, 5]


def test_no_hits_gives_empty_list():
    assert fuse([], [], **DEFAULTS) == []


@pytest.mark.parametrize(("comment_weight", "music_weight"), [(0, 0), (-1, 2), (1, -0.5)])
def test_invalid_weights_are_rejected(comment_weight, music_weight):
    with pytest.raises(ValueError):
        fuse(
            [hit(1, 0.9)],
            [],
            comment_weight=comment_weight,
            music_weight=music_weight,
            min_similarity=0.4,
        )


# ---- 조건 점수와 합치기 -------------------------------------------------------


def _cond(record_id, score, matched=("가을",), missed=()):
    return RecordScore(record_id, score, tuple(matched), tuple(missed))


def _vector(record_id, score):
    return FusedRecord(record_id, score, None, None)


def test_records_are_split_into_exact_and_suggestions():
    ranking = combine_with_conditions(
        [_vector(1, 0.5), _vector(2, 0.5)],
        {
            1: _cond(1, 1.0, matched=("가을", "서울숲"), missed=()),
            2: _cond(2, 0.5, matched=("가을",), missed=("서울숲",)),
        },
        condition_weight=0.5,
    )

    assert [r.record_id for r in ranking.exact] == [1]
    assert [r.record_id for r in ranking.suggestions] == [2]
    assert ranking.suggestions[0].condition.missed == ("서울숲",)


def test_final_score_mixes_vector_and_condition_scores():
    ranking = combine_with_conditions([_vector(1, 0.8)], {1: _cond(1, 0.6)}, condition_weight=0.25)

    (only,) = ranking.exact
    assert math.isclose(only.score, 0.75 * 0.8 + 0.25 * 0.6)
    assert math.isclose(only.vector_score, 0.8)


def test_record_found_only_by_conditions_is_kept_with_zero_vector_score():
    ranking = combine_with_conditions([], {7: _cond(7, 1.0)}, condition_weight=0.5)

    (only,) = ranking.exact
    assert only.record_id == 7 and only.vector_score == 0.0
    assert math.isclose(only.score, 0.5)


def test_record_matching_no_condition_is_left_out_even_with_high_vector_score():
    ranking = combine_with_conditions(
        [_vector(1, 0.99)],
        {1: _cond(1, 0.0, matched=(), missed=("가을",))},
        condition_weight=0.5,
    )

    assert ranking.exact == [] and ranking.suggestions == []


def test_vector_hit_without_condition_data_is_left_out():
    # 최근 N개 읽기 범위 밖의 오래된 기록 등 — 조건을 비교할 수 없으면 결과에 넣지 않는다
    ranking = combine_with_conditions([_vector(9, 0.9)], {}, condition_weight=0.5)

    assert ranking.exact == [] and ranking.suggestions == []


def test_ranking_is_by_final_score_then_newer_record_first():
    ranking = combine_with_conditions(
        [_vector(1, 0.2), _vector(2, 0.9), _vector(3, 0.9), _vector(4, 0.9)],
        {
            1: _cond(1, 1.0, matched=("가을", "서울숲")),
            2: _cond(2, 1.0, matched=("가을", "서울숲")),
            3: _cond(3, 1.0, matched=("가을", "서울숲")),
            4: _cond(4, 0.5, matched=("가을",), missed=("서울숲",)),
        },
        condition_weight=0.5,
    )

    assert [r.record_id for r in ranking.exact] == [3, 2, 1]  # 2·3은 같은 점수 → 3이 최근
    assert [r.record_id for r in ranking.suggestions] == [4]


@pytest.mark.parametrize("weight", [-0.1, 1.1])
def test_invalid_condition_weight_is_rejected(weight):
    with pytest.raises(ValueError):
        combine_with_conditions([], {}, condition_weight=weight)
