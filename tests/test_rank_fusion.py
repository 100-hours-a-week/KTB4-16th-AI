import math

import pytest

from app.components.rank_fusion import fuse
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
