"""RECAP 프롬프트 비교 도구(scripts/recap_distill/compare_prompts.py) — API는 부르지 않는다."""

import json
from collections import Counter

import httpx
import pytest

from scripts.recap_distill import compare_prompts as cp
from scripts.recap_distill.prompt_v7 import V7_SYSTEM_PROMPT as V7_PROMPT
from scripts.recap_distill.prompt_v8 import V8_SYSTEM_PROMPT
from scripts.recap_distill.prompt_v9 import V9_SYSTEM_PROMPT
from scripts.recap_distill.prompt_v10 import (
    ENDING_SECTION,
    ENDINGS,
    V10_SYSTEM_PROMPT,
    ending_section,
)

PROMPT = (
    "[이달의 통계]\n가장 많이 들은 아티스트 아이유, 가장 많이 간 장소 성수동\n\n"
    "[이달의 기록]\n1. 곡 분위기: 잔잔한 발라드 / 아티스트: 아이유 / 장소: 성수동 / 날씨: 비\n"
)
THREE = (
    "당신은 비가 내릴 때마다 잔잔한 노래에 마음을 기대었어요. 성수동에서 보낸 시간이 많았어요. "
    "노래와 함께 마음을 가라앉히던 한 달이었어요."
)


def row(pid, version, text, violations=(), stop="end_turn", out=100):
    return {
        "id": pid,
        "version": version,
        "text": text,
        "stop_reason": stop,
        "usage": {"input": 1000, "output": out},
        "violations": list(violations),
    }


# ---- 프롬프트 ----


def test_v8_keeps_the_v7_data_description_and_changes_only_the_rules():
    head = V7_PROMPT.split("\n- 결과는")[0]
    assert V8_SYSTEM_PROMPT.startswith(head)
    assert "1~3문장" in V8_SYSTEM_PROMPT and "300자" in V8_SYSTEM_PROMPT
    assert "한 문장만 쓴다" not in V8_SYSTEM_PROMPT


def test_v8_examples_use_placeholders_instead_of_concrete_values():
    examples = V8_SYSTEM_PROMPT.split("예시 1:")[1]
    for value in ("진눈깨비", "아이유", "성수동", "Adele", "반포동", "The Weeknd"):
        assert value not in examples
    assert "(아티스트)" in examples and "(장소)" in examples


# ---- 채점 ----


def test_sentence_count_splits_on_sentence_endings_only():
    assert cp.sentence_count("당신은 비를 좋아했어요.") == 1
    assert cp.sentence_count("당신은 비를 좋아했어요") == 1
    assert cp.sentence_count(THREE) == 3
    assert cp.sentence_count("당신은 1.5도의 날을 좋아했어요.") == 1


def test_length_and_sentence_limits_differ_by_version():
    assert cp.violations_for("v8", THREE, PROMPT) == []
    v7 = cp.violations_for("v7", THREE, PROMPT)
    assert "1문장 초과" in v7  # 3문장이라 현재 서비스(v7) 기준으로는 위반
    long_text = "당신은 " + "비가 내렸어요 " * 60
    assert "300자 초과" in cp.violations_for("v8", long_text, PROMPT)
    assert "100자 초과" in cp.violations_for("v7", "당신은 " + "가" * 100, PROMPT)


def test_four_sentences_break_the_v8_limit():
    four = THREE + " 그리고 이어졌어요."
    assert "3문장 초과" in cp.violations_for("v8", four, PROMPT)


def test_emotion_claims_are_flagged_unless_the_input_has_the_same_word():
    text = "당신은 비 오는 날 외로웠던 한 달을 보냈어요."
    assert any("감정을 단정" in v for v in cp.violations_for("v8", text, PROMPT))
    assert not any("감정을 단정" in v for v in cp.violations_for("v8", text, PROMPT + "외로웠"))


def test_existing_rubric_rules_still_apply_to_v8():
    assert "기분점수를 언급함" in cp.violations_for("v8", "당신은 기분점수가 높았어요.", PROMPT)
    assert "'당신은'으로 시작하지 않음" in cp.violations_for("v8", "이번 달은 비가 왔어요.", PROMPT)


# ---- 요약 ----


def test_metrics_counts_commas_per_sentence_and_opening_variety():
    rows = [
        row("a", "v8", "당신은 비가 오면 잔잔한 곡을 들었어요, 성수동에서요. 한 달이었어요."),
        row("b", "v8", "당신은 비가 오면 잔잔한 곡을 들었어요. 두 번째 문장이에요.", ["x"]),
        row("c", "v8", "당신은 눈이 오면 조용한 곡을 들었어요.", [], "max_tokens", 400),
    ]
    summary = cp.metrics(rows)
    assert summary["n"] == 3
    assert summary["commas_per_sentence"] == 1 / 5
    assert summary["opening_variety"] == 2 / 3  # 앞 12글자가 같은 글이 둘
    assert summary["with_violation"] == 1
    assert summary["truncated"] == 1
    assert summary["output_tokens_max"] == 400


def test_most_shared_phrase_finds_copied_wording_anywhere_in_the_text():
    texts = [
        "당신은 비 오는 날 노래를 곁에 두었고 천천히 건너온 한 달이었어요.",
        "당신은 눈 오는 날 음악을 곁에 두었고 조용히 건너온 한 달이었어요.",
        "당신은 맑은 날 산책을 즐겼고 가볍게 지나온 한 달이었어요.",
    ]
    phrase, ratio = cp.most_shared_phrase(texts)
    assert ratio == 1.0 and phrase in texts[0]
    assert cp.most_shared_phrase(["당신은 가나다라마바사.", "당신은 아자차카타파하."])[1] == 0.5
    assert cp.most_shared_phrase([]) == ("", 0.0)


# ---- 사람이 고르는 화면 ----


def test_blind_order_is_deterministic_and_balanced():
    ids = [f"syn-{i:05d}" for i in range(30)]
    order = cp.blind_order(ids, 1)
    assert order == cp.blind_order(ids, 1)
    assert set(order) == set(ids)
    assert Counter(order.values()) == {"v7": 15, "v8": 15}
    assert order != cp.blind_order(ids, 2)


def test_review_data_pairs_only_inputs_that_have_both_versions():
    inputs = [{"id": "a", "prompt": "p-a"}, {"id": "b", "prompt": "p-b"}]
    rows = [row("a", "v7", "A-v7"), row("a", "v8", "A-v8"), row("b", "v7", "B-v7")]
    items, a_is = cp.build_review_data(inputs, rows, 1)
    assert [i["id"] for i in items] == ["a"]
    texts = {"v7": "A-v7", "v8": "A-v8"}
    first = a_is["a"]
    second = "v8" if first == "v7" else "v7"
    assert (items[0]["a"], items[0]["b"]) == (texts[first], texts[second])


def test_long_input_is_cut_in_the_review_page():
    inputs = [{"id": "a", "prompt": "가" * 2000}]
    rows = [row("a", "v7", "x"), row("a", "v8", "y")]
    items, _ = cp.build_review_data(inputs, rows, 1)
    assert len(items[0]["input"]) < 800 and items[0]["input"].endswith("(이하 생략)")


# ---- v9 ----


def test_v9_keeps_the_data_description_and_gives_varied_endings():
    head = V7_PROMPT.split("\n- 결과는")[0]
    assert V9_SYSTEM_PROMPT.startswith(head)
    flat = " ".join(V9_SYSTEM_PROMPT.split())  # 프롬프트는 줄바꿈으로 나뉘어 있어서 한 줄로 편다
    assert "여섯 가지" in flat and "같은 끝맺음을 매번 쓰지 않는다" in flat
    assert "입력에 없는 행동·장소·사물을 덧붙이지 않는다" in flat


def test_v9_examples_use_placeholders_and_do_not_end_alike():
    examples = V9_SYSTEM_PROMPT.split("예시 1:")[1].split("- 다정한 톤")[0]
    for value in ("진눈깨비", "아이유", "성수동", "Adele", "반포동", "The Weeknd", "골목"):
        assert value not in examples
    assert "(아티스트)" in examples and "(장소)" in examples
    assert "한 달이었어요" not in examples  # v8 예시의 끝맺음이 모든 글로 번졌던 문제


def test_invented_details_are_flagged_unless_the_input_mentions_them():
    text = "당신은 비 오는 날 골목이 떠오르는 노래를 들었어요."
    assert any("입력에 없는 묘사" in v for v in cp.violations_for("v9", text, PROMPT))
    assert not any("입력에 없는 묘사" in v for v in cp.violations_for("v9", text, PROMPT + "골목"))


def test_v9_uses_the_same_limits_as_v8():
    assert cp.VERSIONS["v9"]["limit"] == cp.VERSIONS["v8"]["limit"] == 300
    assert cp.VERSIONS["v9"]["max_sentences"] == 3
    assert cp.VERSIONS["v9"]["max_tokens"] < cp.VERSIONS["v8"]["max_tokens"]


def test_review_pair_can_be_any_two_versions():
    inputs = [{"id": f"s{i}", "prompt": "p"} for i in range(4)]
    rows = [row(i["id"], v, f"{i['id']}-{v}") for i in inputs for v in ("v7", "v8", "v9")]
    items, a_is = cp.build_review_data(inputs, rows, 1, ("v8", "v9"))
    assert set(a_is.values()) == {"v8", "v9"}
    for item in items:
        first = a_is[item["id"]]
        second = "v9" if first == "v8" else "v8"
        assert (item["a"], item["b"]) == (f"{item['id']}-{first}", f"{item['id']}-{second}")
    only_v7_v8 = [r for r in rows if r["version"] != "v9"]
    assert cp.build_review_data(inputs, only_v7_v8, 1, ("v8", "v9"))[0] == []


def test_stray_english_reasoning_is_flagged_but_input_names_are_not():
    leaked = "당신은 비를 좋아했어요.\n\nWait, I must avoid inventing details."
    assert any("영어가 섞임" in v for v in cp.violations_for("v9", leaked, PROMPT))
    named = "당신은 Daniel Caesar의 노래를 자주 들었어요."
    assert cp.violations_for("v9", named, PROMPT + "Daniel Caesar") == []


# ---- v10 ----


def test_v10_drops_the_candidate_list_and_the_forbidden_example_words():
    flat = " ".join(V10_SYSTEM_PROMPT.split())
    assert ENDING_SECTION in flat and "끝맺음 부분은 바꾸지 않는다" in flat
    assert "여섯 가지" not in flat and "골목" not in flat
    assert "하루하루" not in flat  # v9 예시에서 모델이 베낀 구절
    for _, key in ENDINGS:  # 후보 목록은 프롬프트가 아니라 코드가 들고 있다
        assert f") {key}" not in flat


def test_ending_section_is_a_single_labelled_line():
    assert ending_section("~하던 날들이었어요") == "[이번 마무리 끝맺음]\n~하던 날들이었어요"


def test_assign_endings_is_balanced_and_deterministic():
    ids = [f"syn-{i:05d}" for i in range(30)]
    first = cp.assign_endings(ids, 1)
    assert first == cp.assign_endings(ids, 1)
    assert first != cp.assign_endings(ids, 2)
    counts = Counter(first.values())
    assert set(counts) == {ending for ending, _ in ENDINGS}
    assert set(counts.values()) == {5}


def test_missing_the_assigned_ending_is_flagged():
    ending, key = ENDINGS[0]
    followed = f"당신은 비를 좋아했어요. 비 오는 날을 {key}."
    ignored = "당신은 비를 좋아했어요. 한 달이었어요."
    assert cp.violations_for("v10", followed, PROMPT, ending) == []
    assert "지정한 끝맺음을 쓰지 않음" in cp.violations_for("v10", ignored, PROMPT, ending)
    assert "지정한 끝맺음을 쓰지 않음" not in cp.violations_for("v10", ignored, PROMPT)


def test_metrics_reports_ending_variety_and_following():
    ending, key = ENDINGS[0]
    rows = [
        row("a", "v10", f"당신은 비를 좋아했어요. 그 {key}.", [], "end_turn") | {"ending": ending},
        row("b", "v10", "당신은 눈을 좋아했어요. 한 달이었어요.", ["지정한 끝맺음을 쓰지 않음"])
        | {"ending": ending},
    ]
    summary = cp.metrics(rows)
    assert summary["ending_followed"] == 1
    assert summary["ending_variety"] == 1.0 and summary["top_ending_ratio"] == 0.5
    assert cp.metrics([row("c", "v9", "당신은 비를 좋아했어요.")])["ending_followed"] is None


def test_shared_phrase_ignores_the_gap_between_sentences():
    # 두 글이 공유하는 6글자 조각은 "어요. 사진"뿐이고, 이건 문장이 이어지는 자리라서 세지 않는다
    texts = [
        "당신은 가나다라마바어요. 사진에는 하늘이었어요.",
        "당신은 오른쪽위아래어요. 사진에는 바다였어요.",
    ]
    phrase, ratio = cp.most_shared_phrase(texts)
    assert ratio == 0.5 and ". " not in phrase


def test_missing_final_period_and_broken_ending_are_flagged():
    assert "마지막에 마침표가 없음" in cp.violations_for("v10", "당신은 비를 좋아했어요", PROMPT)
    assert "마지막에 마침표가 없음" not in cp.violations_for(
        "v10", "당신은 비를 좋아했어요.", PROMPT
    )
    broken = "당신은 천천히 걸어 이 자리까지 와 있었어요을 지나 여기까지 왔어요."
    assert "끝맺음 뒤에 조사가 붙어 문장이 깨짐" in cp.violations_for("v10", broken, PROMPT)
    fine = "당신은 비를 좋아했어요. 사진에는 바다가 담겼어요."
    assert cp.violations_for("v10", fine, PROMPT) == []


def _vllm_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _chat(client, **kw):
    args = {
        "base_url": "https://pod-8000.proxy.runpod.net/v1/",
        "api_key": "test-key",
        "model": "gemma-3-12b",
        "system": "시스템",
        "prompt": "입력",
        "max_tokens": 300,
    }
    return cp.vllm_chat(client, **{**args, **kw})


def test_vllm_chat_sends_openai_request_and_parses_the_answer():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["agent"] = request.headers["user-agent"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": " 당신은 비를 좋아했어요. "}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1200, "completion_tokens": 55},
            },
        )

    text, stop, tokens_in, tokens_out = _chat(_vllm_client(handler))

    assert seen["url"] == "https://pod-8000.proxy.runpod.net/v1/chat/completions"
    assert seen["auth"] == "Bearer test-key"
    assert seen["agent"] == cp.VLLM_USER_AGENT and "python" not in seen["agent"].lower()
    assert seen["body"]["model"] == "gemma-3-12b" and seen["body"]["max_tokens"] == 300
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "시스템"},
        {"role": "user", "content": "입력"},
    ]
    assert "temperature" not in seen["body"]  # 샘플링 값은 서버 기본을 따른다
    assert (text, stop, tokens_in, tokens_out) == ("당신은 비를 좋아했어요.", "stop", 1200, 55)


def test_vllm_chat_maps_length_stop_to_max_tokens():
    def handler(request):
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "당신은 비"}, "finish_reason": "length"}]},
        )

    assert _chat(_vllm_client(handler))[1] == "max_tokens"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(502, text="Bad Gateway"),
        httpx.Response(401, text="Unauthorized"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, text="not json"),
    ],
)
def test_vllm_chat_failures_stop_the_experiment(response):
    with pytest.raises(cp.CallFailed):
        _chat(_vllm_client(lambda request: response))


def test_vllm_chat_connection_error_stops_the_experiment():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(cp.CallFailed, match="서버 연결 실패"):
        _chat(_vllm_client(handler))


def test_output_path_does_not_nest_directories_for_org_model_names():
    path = cp.output_path("gaunernst/gemma-3-12b-it-int4-awq")
    assert path.parent == cp.OUT_DIR
    assert path.name == "outputs_gaunernst_gemma-3-12b-it-int4-awq.jsonl"
    assert cp.output_path("claude-haiku-5-5").name == "outputs_claude-haiku-5-5.jsonl"


def test_final_period_is_added_only_to_finished_texts_without_punctuation():
    assert cp.with_final_period("당신은 비를 좋아했어요", "stop") == "당신은 비를 좋아했어요."
    assert cp.with_final_period("당신은 비를 좋아했어요.", "stop") == "당신은 비를 좋아했어요."
    assert cp.with_final_period("당신은 비를 좋아했어요…", "end_turn") == "당신은 비를 좋아했어요…"
    assert (
        cp.with_final_period("  당신은 비를  ", "max_tokens") == "당신은 비를"
    )  # 잘린 글은 그대로
    assert cp.with_final_period("", "stop") == ""


def test_relabel_rows_keeps_one_version_and_fixes_the_period_for_both_models():
    rows = [
        {"id": "a", "version": "v9", "text": "당신은 비를 좋아했어요", "stop_reason": "stop"},
        {"id": "a", "version": "v8", "text": "다른 버전", "stop_reason": "stop"},
        {"id": "z", "version": "v9", "text": "대상이 아닌 입력", "stop_reason": "stop"},
    ]

    out = cp.relabel_rows(rows, "gemma", "v9", {"a"})

    assert [(r["id"], r["version"], r["text"]) for r in out] == [
        ("a", "gemma v9", "당신은 비를 좋아했어요.")
    ]


def test_cross_model_review_pairs_the_two_models_blind():
    ids = ["a", "b", "c", "d"]
    inputs = [{"id": i, "prompt": PROMPT} for i in ids]
    rows = []
    for label, text in (("gemma", "젬마 글"), ("haiku", "하이쿠 글")):
        raw = [
            {"id": i, "version": "v9", "text": f"{text} {i}.", "stop_reason": "stop"} for i in ids
        ]
        rows += cp.relabel_rows(raw, label, "v9", set(ids))

    items, a_is = cp.build_review_data(inputs, rows, 1, ("gemma v9", "haiku v9"))

    assert len(items) == 4
    assert set(a_is.values()) == {"gemma v9", "haiku v9"}
    for item in items:
        gemma_is_a = a_is[item["id"]] == "gemma v9"
        assert item["a"].startswith("젬마" if gemma_is_a else "하이쿠")
        assert item["b"].startswith("하이쿠" if gemma_is_a else "젬마")
