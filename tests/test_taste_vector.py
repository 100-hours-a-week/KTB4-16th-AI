import math

from app.components.taste_vector import blend_taste, mean_vector


def _length(v):
    return math.sqrt(sum(x * x for x in v))


def test_mean_vector_averages_and_normalizes():
    mean = mean_vector([[1.0, 0.0], [0.0, 1.0]])

    assert mean is not None
    assert abs(mean[0] - mean[1]) < 1e-9 and abs(_length(mean) - 1.0) < 1e-9


def test_mean_vector_is_none_for_empty_or_mixed_lengths():
    assert mean_vector([]) is None
    assert mean_vector([[1.0, 0.0], [1.0, 0.0, 0.0]]) is None


def test_weight_decides_how_far_the_query_moves_toward_taste():
    situation, taste = [1.0, 0.0], [0.0, 1.0]

    low = blend_taste(situation, taste, 0.3)
    high = blend_taste(situation, taste, 0.7)

    assert low[0] > low[1]  # 상황 70% — 아직 상황 쪽
    assert high[1] > high[0]  # 취향 70% — 취향 쪽으로 넘어감
    assert abs(_length(low) - 1.0) < 1e-9


def test_no_blend_when_off_missing_or_dimension_differs():
    situation = [1.0, 0.0]

    assert blend_taste(situation, [0.0, 1.0], 0.0) == situation
    assert blend_taste(situation, None, 0.3) == situation
    assert blend_taste(situation, [0.0, 1.0, 0.0], 0.3) == situation
