import pytest

from scripts.photo_eval import score_descriptions as sd

VOCAB = {
    "장소": ["바다", "산"],
    "사진 종류": ["진짜 사진", "그림·짤"],
    "대상": ["풍경·사물"],
    "분위기": ["푸르고 밝은"],
    "표정·감정": ["웃음·신남"],
    "노래 분위기": ["잔잔한", "신나는", "차분한", "웅장한"],
}


def test_f1_both_empty_is_perfect_and_one_empty_is_zero():
    assert sd.f1(set(), set()) == 1.0
    assert sd.f1({"a"}, set()) == 0.0
    assert sd.f1(set(), {"a"}) == 0.0


def test_f1_partial_overlap():
    # 맞힌 1개, 예측 2개, 정답 2개 → 정밀도 0.5, 재현율 0.5
    assert sd.f1({"a", "b"}, {"a", "c"}) == pytest.approx(0.5)
    assert sd.f1({"a"}, {"a"}) == 1.0


def test_info_score_averages_over_all_attributes():
    gold = {a: [] for a in sd.ATTRS}
    pred = {a: [] for a in sd.ATTRS}
    score, per_attr = sd.info_score(pred, gold)
    assert score == 1.0 and set(per_attr) == set(sd.ATTRS)

    pred["장소"] = ["바다"]  # 정답은 비어 있는데 하나를 고름 → 그 속성만 0점
    score, per_attr = sd.info_score(pred, gold)
    assert per_attr["장소"] == 0.0
    assert score == pytest.approx((len(sd.ATTRS) - 1) / len(sd.ATTRS))


def test_parse_json_handles_fences_and_surrounding_text():
    assert sd.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert sd.parse_json('설명입니다 {"a": [1, 2]} 끝') == {"a": [1, 2]}
    assert sd.parse_json("JSON 없음") is None
    assert sd.parse_json("{깨진 json}") is None
    assert sd.parse_json("[1, 2]") is None


def test_clean_attributes_drops_unknown_values_and_trims_to_limit():
    raw = {
        "장소": ["바다", "없는곳", "바다", "산"],
        "사진 종류": "진짜 사진",  # 리스트가 아니면 빈 값으로 본다
        "노래 분위기": ["잔잔한", "신나는", "차분한", "웅장한"],
    }
    cleaned = sd.clean_attributes(raw, VOCAB)
    assert cleaned["장소"] == ["바다", "산"]
    assert cleaned["사진 종류"] == []
    assert cleaned["노래 분위기"] == ["잔잔한", "신나는", "차분한"]
    assert cleaned["대상"] == []


def test_count_claims_counts_verdicts_and_serious_only_for_unsupported():
    raw = {
        "claims": [
            {"claim": "a", "verdict": "supported", "serious": True},
            {"claim": "b", "verdict": "unsupported", "serious": True},
            {"claim": "c", "verdict": "unsupported", "serious": False},
            {"claim": "d", "verdict": "unverifiable"},
            {"claim": "e", "verdict": "모름"},
        ]
    }
    counts = sd.count_claims(raw)
    assert (counts["supported"], counts["unsupported"], counts["unverifiable"]) == (1, 2, 1)
    assert counts["serious"] == 1
    assert len(counts["claims"]) == 4
    assert sd.count_claims({"claims": "x"}) is None


def test_claim_precision_ignores_unverifiable():
    claims = {"supported": 3, "unsupported": 1, "unverifiable": 5}
    assert sd.claim_precision(claims) == 0.75
    assert sd.claim_precision({"supported": 0, "unsupported": 0, "unverifiable": 2}) is None
    assert sd.claim_precision(None) is None


def test_shifted_never_maps_to_itself_and_is_a_permutation():
    ids = [f"p{i}" for i in range(47)]
    mapping = sd.shifted(ids)
    assert all(pid != other for pid, other in mapping.items())
    assert sorted(mapping.values()) == sorted(ids)


def test_sign_test_matches_known_values():
    assert sd.sign_test_p(8, 6) == pytest.approx(0.7905, abs=1e-3)
    assert sd.sign_test_p(0, 0) == 1.0
    assert sd.sign_test_p(10, 0) == pytest.approx(2 / 2**10)


def test_bootstrap_ci_is_deterministic_and_brackets_the_mean():
    diffs = [0.1, 0.2, 0.0, 0.3, 0.1, 0.2, 0.15, 0.05]
    low, high = sd.bootstrap_ci(diffs)
    assert (low, high) == sd.bootstrap_ci(diffs)
    mean = sum(diffs) / len(diffs)
    assert low <= mean <= high


def test_paired_compare_counts_wins_losses_and_ties():
    base = {"a": 0.5, "b": 0.5, "c": 0.5, "d": 0.2}
    new = {"a": 0.7, "b": 0.5, "c": 0.4, "x": 1.0}  # d와 x는 짝이 없어 뺀다
    result = sd.paired_compare(base, new)
    assert result["n"] == 3
    assert (result["wins"], result["ties"], result["losses"]) == (1, 1, 1)
    assert result["diff"] == pytest.approx((0.2 + 0.0 - 0.1) / 3)
    assert result["relative"] == pytest.approx(result["diff"] / 0.5)
    assert sd.paired_compare({}, {"a": 1.0}) is None


def test_summarize_pools_claims_and_reports_control_adjusted_score():
    def item(pid, info, control, supported, unsupported, serious):
        return {
            "id": pid,
            "split": "tune",
            "chars": 100,
            "info": info,
            "attr_f1": {a: info for a in sd.ATTRS},
            "control_info": control,
            "supported": supported,
            "unsupported": unsupported,
            "serious": serious,
        }

    results = [item("a", 0.8, 0.4, 3, 1, 1), item("b", 0.6, 0.2, 5, 0, 0)]
    summary = sd.summarize(results)
    assert summary["info"] == pytest.approx(0.7)
    assert summary["control"] == pytest.approx(0.3)
    assert summary["info_over_control"] == pytest.approx((0.7 - 0.3) / (1 - 0.3))
    assert summary["precision"] == pytest.approx(8 / 9)  # 설명마다가 아니라 주장 전체로 합친다
    assert summary["serious_descriptions"] == 1
    assert summary["wrong_per_description"] == 0.5


def test_manual_agreement_counts_both_directions():
    results = [
        {"id": "a", "supported": 3, "unsupported": 1, "serious": 1},
        {"id": "b", "supported": 4, "unsupported": 1, "serious": 0},
        {"id": "c", "supported": 5, "unsupported": 0, "serious": 0},
        {"id": "d", "supported": None, "unsupported": None, "serious": None},
    ]
    manual = {
        "a": {"level": "중대"},
        "b": {"level": "없음"},
        "c": {"level": "사소"},
        "d": {"level": "중대"},
    }
    agreement = sd.manual_agreement(results, manual)
    # 큰 오류: a(둘 다 있음), b(둘 다 없음), c(둘 다 없음) → 3개 모두 같음. 판정이 없는 d는 뺀다
    assert agreement["serious"] == {
        "agree": 3,
        "n": 3,
        "both_yes": 1,
        "claude_only": 0,
        "manual_only": 0,
    }
    # 틀린 주장: a 일치, b는 Claude만 있음, c는 내 판정만 있음
    assert agreement["any_wrong"]["agree"] == 1
    assert agreement["any_wrong"]["claude_only"] == 1
    assert agreement["any_wrong"]["manual_only"] == 1
