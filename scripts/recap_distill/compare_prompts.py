"""RECAP 요약 프롬프트 v7(서비스 현재)과 후보 v8·v9(300자용)를 같은 가짜 입력으로 비교한다.

세 단계다.
- run: 가짜 입력마다 고른 버전으로 모델을 한 번씩 불러 결과를 저장한다. 기본은 Claude(Haiku 5.5)이고
  외부 API라 비용이 들어서 --budget-usd 한도, --dry-run(건수·예상 비용만 보기), 이어하기를 둔다.
  입력은 전부 가짜 데이터(synth_inputs.py)이고 실제 사용자 기록은 보내지 않는다.
  API 키는 앱 설정(.env의 ANTHROPIC_API_KEY)에서 읽고 출력하지 않는다.
  --backend vllm이면 RunPod 등에서 서빙하는 vLLM(OpenAI 호환 주소)을 부른다. 이때 --model은
  서빙 이름이고, 주소는 --base-url(또는 환경변수 VLLM_BASE_URL), 서버 키는 환경변수
  VLLM_API_KEY에서만 읽는다(출력하지 않음). 비용 계산은 하지 않는다. 같은 서빙 이름에
  가중치만 바뀌는 경우(FP8→AWQ 등)는 --label로 결과 파일을 따로 정한다.
- report: 저장된 결과를 자동으로 재서 표로 보여 주고, 사람이 읽고 고르는 화면(review.html)을
  만든다(--pair로 비교할 두 버전을 고른다). 화면에서는 어느 쪽이 어느 버전인지 가려 둔다.
- 자동으로 재는 것은 규칙 위반, 길이, 문장 수, 쉼표 수(나열 정도), 시작 표현의 다양성뿐이다.
  "감성적인가", "나열식이 줄었는가"는 숫자로 못 재서 화면에서 사람이 고른다.

실행 예:
    PYTHONPATH=. .venv/bin/python scripts/recap_distill/compare_prompts.py run --dry-run
    PYTHONPATH=. .venv/bin/python scripts/recap_distill/compare_prompts.py run --versions v9
    PYTHONPATH=. .venv/bin/python scripts/recap_distill/compare_prompts.py report --pair v8 v9
    PYTHONPATH=. .venv/bin/python scripts/recap_distill/compare_prompts.py run --backend vllm \\
        --base-url https://<PodID>-8000.proxy.runpod.net/v1 --model <서빙 이름> --versions v9
    PYTHONPATH=. .venv/bin/python scripts/recap_distill/compare_prompts.py compare-models \\
        --a gemma-3-12b_awq --b claude-haiku-5-5 --version v9
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import anthropic
import httpx

from app.config import get_settings
from scripts.recap_distill.generate_labels import PRICES, cost_of
from scripts.recap_distill.prompt_v7 import V7_SYSTEM_PROMPT
from scripts.recap_distill.prompt_v8 import V8_MAX_SENTENCES, V8_SYSTEM_PROMPT, V8_TEXT_LIMIT
from scripts.recap_distill.prompt_v9 import V9_MAX_SENTENCES, V9_SYSTEM_PROMPT, V9_TEXT_LIMIT
from scripts.recap_distill.prompt_v10 import (
    ENDINGS,
    V10_MAX_SENTENCES,
    V10_SYSTEM_PROMPT,
    V10_TEXT_LIMIT,
    ending_section,
)
from scripts.recap_distill.rubric import check
from scripts.recap_distill.synth_inputs import build_one

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data/recap_distill/prompt_compare"
TEMPLATE = Path(__file__).with_name("compare_template.html")

# v7은 서비스와 같은 값(max_tokens 150, 100자, 한 문장). v8은 300자까지 쓰게 하므로 출력 여유를
# 넉넉히 준다. 실제로 몇 토큰이 필요한지는 이 실험의 출력 토큰 수로 확인한다.
VERSIONS = {
    "v7": {"system": V7_SYSTEM_PROMPT, "max_tokens": 150, "limit": 100, "max_sentences": 1},
    "v8": {
        "system": V8_SYSTEM_PROMPT,
        "max_tokens": 400,
        "limit": V8_TEXT_LIMIT,
        "max_sentences": V8_MAX_SENTENCES,
    },
    # v8 실험에서 최대 출력이 193토큰이었다. 여유를 두고 300으로 줄여 같은 조건에서 잘리는지 본다.
    "v9": {
        "system": V9_SYSTEM_PROMPT,
        "max_tokens": 300,
        "limit": V9_TEXT_LIMIT,
        "max_sentences": V9_MAX_SENTENCES,
    },
    # v10은 코드가 끝맺음을 뽑아 입력 끝에 붙인다("uses_ending").
    "v10": {
        "system": V10_SYSTEM_PROMPT,
        "max_tokens": 300,
        "limit": V10_TEXT_LIMIT,
        "max_sentences": V10_MAX_SENTENCES,
        "uses_ending": True,
    },
}
ENDING_KEYS = dict(ENDINGS)  # 입력에 넣은 끝맺음 → 글에 들어 있어야 하는 고정 부분
# 이 두 항목은 버전마다 기준이 달라서 check()의 결과에서 빼고 아래에서 다시 판정한다.
VERSION_DEPENDENT_RULES = ("100자 초과", "한 문장이 아님")
# 사용자의 감정을 단정하는 말. 입력(곡 분위기 글 등)에 같은 말이 있으면 위반으로 보지 않는다.
EMOTION_CLAIMS = ("외로웠", "행복했", "위로받", "우울했", "슬펐", "지쳤", "힘들었")
# 입력에 없는데 모델이 덧붙인 행동·장소 묘사. 단어 목록으로 찾는 것이라 일부만 잡는다.
INVENTED_DETAILS = ("골목", "산길", "따라 불", "걷", "걸으", "거닐", "산책", "익숙한")
# 문장이 끝난 자리("~어요") 바로 뒤에 조사가 붙은 깨진 문장. 끝맺음을 억지로 이어 붙일 때 생긴다.
BROKEN_ENDING = re.compile(r"(어요|았어요|였어요|했어요|왔어요|갔어요)[을를이가은는와과로에]")
EST_TOKENS_PER_CHAR = 1.1  # generate_labels.py와 같은 사전 추정값
VLLM_USER_AGENT = "muro-recap-eval/1.0"  # RunPod 프록시는 파이썬 기본 User-Agent를 막는다
VLLM_TIMEOUT_SECONDS = 120.0  # 처음 한 번은 모델이 깨어나는 데 오래 걸릴 수 있다
EST_OUTPUT_TOKENS = {"v7": 100, "v8": 300, "v9": 300, "v10": 300}
PROMPT_PREVIEW_CHARS = 700


def sentence_count(text: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?…])\s+", text.strip()) if part])


def last_sentence(text: str) -> str:
    parts = [p for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p]
    return parts[-1] if parts else ""


def assign_endings(ids: list[str], seed: int) -> dict[str, str]:
    """입력마다 끝맺음 하나를 정한다. 실험에서는 후보가 고르게 나오도록 섞어서 나눈다.

    서비스에서는 사용자 번호와 연월로 뽑으면 되고, 그러면 같은 달을 다시 만들어도 같다.
    """
    pool = [ending for ending, _ in ENDINGS]
    picks = [pool[i % len(pool)] for i in range(len(ids))]
    random.Random(f"{seed}-ending").shuffle(picks)
    return dict(zip(sorted(ids), picks, strict=True))


def violations_for(version: str, text: str, prompt: str, ending: str | None = None) -> list[str]:
    spec = VERSIONS[version]
    found = [v for v in check(text, prompt) if v not in VERSION_DEPENDENT_RULES]
    if len(text) > spec["limit"]:
        found.append(f"{spec['limit']}자 초과")
    if sentence_count(text) > spec["max_sentences"]:
        found.append(f"{spec['max_sentences']}문장 초과")
    if "\n" in text:
        found.append("줄바꿈이 있음")
    claimed = [word for word in EMOTION_CLAIMS if word in text and word not in prompt]
    if claimed:
        found.append(f"사용자 감정을 단정함: {', '.join(claimed)}")
    # 모델이 "고쳐 쓰겠다" 같은 혼잣말을 영어로 글에 흘린 경우를 찾는다. 입력에 있는 영어(가수
    # 이름 등)는 제외한다.
    stray = [w for w in re.findall(r"[A-Za-z]{3,}", text) if w not in prompt]
    if stray:
        found.append(f"입력에 없는 영어가 섞임: {', '.join(stray[:3])}")
    invented = [word for word in INVENTED_DETAILS if word in text and word not in prompt]
    if invented:
        found.append(f"입력에 없는 묘사: {', '.join(invented)}")
    if text and not re.search(r"[.!?…]$", text.strip()):
        found.append("마지막에 마침표가 없음")
    if BROKEN_ENDING.search(text):
        found.append("끝맺음 뒤에 조사가 붙어 문장이 깨짐")
    if ending and ENDING_KEYS[ending] not in last_sentence(text):
        found.append("지정한 끝맺음을 쓰지 않음")
    return found


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def output_path(model: str) -> Path:
    # 서빙 이름에는 "org/model"처럼 슬래시가 들어갈 수 있어서 파일 이름에는 쓰지 않는다.
    return OUT_DIR / f"outputs_{model.replace('/', '_')}.jsonl"


class CallFailed(Exception):
    """모델 호출이 실패해 실험을 멈춰야 할 때."""


def vllm_chat(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    prompt: str,
    max_tokens: int,
) -> tuple[str, str, int, int]:
    """vLLM의 OpenAI 호환 채팅 주소를 한 번 불러 (글, 정지 사유, 입력 토큰, 출력 토큰)을 돌려준다.

    정지 사유는 Anthropic 쪽과 맞춰 "length"를 "max_tokens"로 바꾼다. 온도 등 샘플링 값은
    보내지 않아서 서버(모델 저장소의 generation_config)의 기본값을 따른다.
    """
    try:
        res = client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "User-Agent": VLLM_USER_AGENT},
            json={
                "model": model,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
            },
        )
    except httpx.TransportError as e:
        raise CallFailed(f"서버 연결 실패 ({type(e).__name__})") from e
    if res.status_code != 200:
        raise CallFailed(f"서버 오류 (status {res.status_code}) {res.text[:200]}")
    try:
        body = res.json()
        choice = body["choices"][0]
        text = (choice["message"].get("content") or "").strip()
    except (ValueError, KeyError, IndexError, AttributeError) as e:
        raise CallFailed("서버 응답 형식이 예상과 달라요") from e
    finish = choice.get("finish_reason") or ""
    usage = body.get("usage") or {}
    return (
        text,
        "max_tokens" if finish == "length" else finish,
        int(usage.get("prompt_tokens", 0)),
        int(usage.get("completion_tokens", 0)),
    )


def anthropic_chat(
    client: anthropic.Anthropic, model: str, system: str, prompt: str, max_tokens: int
) -> tuple[str, str, int, int]:
    """Claude를 한 번 불러 vllm_chat과 같은 모양으로 돌려준다."""
    try:
        res = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
            # 서비스(app/clients/anthropic_client.py)와 같은 옵션
            output_config={"effort": "low"},
        )
    except anthropic.APIStatusError as e:
        raise CallFailed(f"API 오류 (status {e.status_code})") from e
    except anthropic.APIConnectionError as e:
        raise CallFailed("API 연결 실패") from e
    text = "".join(b.text for b in res.content if b.type == "text").strip()
    return text, res.stop_reason or "", res.usage.input_tokens, res.usage.output_tokens


def make_inputs(count: int, seed: int) -> list[dict]:
    return [build_one(i, seed) for i in range(count)]


def estimate_cost(model: str, inputs: list[dict], todo: set[tuple[str, str]]) -> float:
    total = 0.0
    for row in inputs:
        for version, spec in VERSIONS.items():
            if (row["id"], version) in todo:
                chars = len(spec["system"]) + len(row["prompt"])
                total += cost_of(
                    model, int(chars * EST_TOKENS_PER_CHAR), EST_OUTPUT_TOKENS[version]
                )
    return total


def run(args: argparse.Namespace) -> None:
    use_vllm = args.backend == "vllm"
    if use_vllm and args.model in PRICES:
        sys.exit("--backend vllm에서는 --model에 서버의 서빙 이름을 적어야 해요.")
    if not use_vllm and args.model not in PRICES:
        sys.exit(f"Claude 모델은 {', '.join(sorted(PRICES))} 중에서 고르세요.")
    out = output_path(args.label or args.model)
    inputs = make_inputs(args.count, args.seed)
    done = {(r["id"], r["version"]) for r in read_jsonl(out)}
    endings = assign_endings([row["id"] for row in inputs], args.seed)
    todo = {(row["id"], v) for row in inputs for v in args.versions} - done
    est = 0.0 if use_vllm else estimate_cost(args.model, inputs, todo)
    print(f"모델 {args.model} ({args.backend}) | 입력 {len(inputs)}건")
    print(f"버전 {', '.join(args.versions)}")
    print(f"이번에 부를 횟수 {len(todo)}회 (이미 저장된 {len(done)}건은 건너뜀)")
    if use_vllm:
        print("비용 계산 없음 (자체 서버). 서버 사용 시간만 든다.")
    else:
        print(f"사전 추정 비용 ${est:.4f} | 한도 ${args.budget_usd:.2f}")
    if args.dry_run or not todo:
        return
    if est > args.budget_usd:
        print("추정 비용이 한도를 넘어서 시작하지 않아요. --count나 --budget-usd를 조정하세요.")
        sys.exit(1)

    if use_vllm:
        base_url = args.base_url or os.environ.get("VLLM_BASE_URL", "")
        api_key = os.environ.get("VLLM_API_KEY", "")
        if not base_url:
            sys.exit(
                "--base-url(또는 환경변수 VLLM_BASE_URL)에 서버 주소를 주세요. 끝은 /v1이에요."
            )
        if not api_key:
            sys.exit("환경변수 VLLM_API_KEY가 없어요.")
        http = httpx.Client(timeout=VLLM_TIMEOUT_SECONDS)

        def generate(spec: dict, user_prompt: str) -> tuple[str, str, int, int]:
            return vllm_chat(
                http, base_url, api_key, args.model, spec["system"], user_prompt, spec["max_tokens"]
            )
    else:
        api_key = get_settings().anthropic_api_key
        if not api_key:
            sys.exit(".env에 ANTHROPIC_API_KEY가 없어요.")
        client = anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=60)

        def generate(spec: dict, user_prompt: str) -> tuple[str, str, int, int]:
            return anthropic_chat(
                client, args.model, spec["system"], user_prompt, spec["max_tokens"]
            )

    spent, calls = 0.0, 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for row in inputs:
            for version, spec in VERSIONS.items():
                if (row["id"], version) not in todo:
                    continue
                average = spent / calls if calls else est / len(todo)
                if spent + average * 1.5 > args.budget_usd:
                    print(f"한도에 가까워서 {calls}회에서 멈춰요 (사용 ${spent:.4f}).")
                    return
                ending = endings[row["id"]] if spec.get("uses_ending") else None
                user_prompt = row["prompt"]
                if ending:
                    user_prompt += "\n\n" + ending_section(ending)
                try:
                    text, stop_reason, tokens_in, tokens_out = generate(spec, user_prompt)
                except CallFailed as e:
                    print(f"{e} — {calls}회에서 멈춰요.")
                    return
                calls += 1
                if not use_vllm:
                    spent += cost_of(args.model, tokens_in, tokens_out)
                if not text:  # 저장하지 않는다 — 다시 실행하면 이 건을 다시 시도한다
                    print(f"{row['id']} {version}: 빈 응답 (정지 사유 {stop_reason}) — 건너뜀")
                    continue
                violations = violations_for(version, text, row["prompt"], ending)
                if stop_reason == "max_tokens":
                    violations.append("max_tokens에서 잘림")
                record = {
                    "id": row["id"],
                    "version": version,
                    "text": text,
                    "model": args.model,
                    "stop_reason": stop_reason,
                    "usage": {"input": tokens_in, "output": tokens_out},
                    "violations": violations,
                }
                if ending:
                    record["ending"] = ending
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
    print(f"이번 실행 {calls}회" + ("" if use_vllm else f", 비용 ${spent:.4f}"))


def most_shared_phrase(texts: list[str], size: int = 6) -> tuple[str, float]:
    """가장 많은 글에 똑같이 들어 있는 `size`글자 표현과, 그 표현이 든 글의 비율.

    모델이 프롬프트 예시를 베껴 모든 글이 비슷해지는지 보려는 것이다. 시작 12글자만 보는
    "시작 표현 다양성"은 글 끝이나 중간의 베끼기를 못 잡는다.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        body = text.removeprefix("당신은").strip()
        grams: set[str] = set()
        for sentence in re.split(r"(?<=[.!?…])\s+", body):  # 문장이 이어지는 자리는 건너뛴다
            grams.update(sentence[i : i + size] for i in range(len(sentence) - size + 1))
        counts.update(grams)
    if not counts:
        return "", 0.0
    phrase, found = counts.most_common(1)[0]
    return phrase, found / len(texts)


def metrics(rows: list[dict]) -> dict:
    """한 버전의 결과 묶음을 숫자로 요약한다."""
    n = len(rows)
    texts = [r["text"] for r in rows]
    sentences = sum(sentence_count(t) for t in texts)
    rules = Counter(v.split(":")[0] for r in rows for v in r["violations"])
    shared_phrase, shared_ratio = most_shared_phrase(texts)
    tails = Counter(re.sub(r"[.\s]+$", "", last_sentence(t))[-7:] for t in texts)
    with_ending = [r for r in rows if r.get("ending")]
    return {
        "ending_variety": len(tails) / n,
        "top_ending_ratio": tails.most_common(1)[0][1] / n,
        "ending_followed": (
            sum("지정한 끝맺음을 쓰지 않음" not in r["violations"] for r in with_ending)
            if with_ending
            else None
        ),
        "shared_phrase": shared_phrase,
        "shared_ratio": shared_ratio,
        "n": n,
        "chars_avg": sum(len(t) for t in texts) / n,
        "chars_max": max(len(t) for t in texts),
        "sentences_avg": sentences / n,
        "commas_per_sentence": sum(t.count(",") for t in texts) / sentences,
        "opening_variety": len({t[:12] for t in texts}) / n,
        "with_violation": sum(bool(r["violations"]) for r in rows),
        "truncated": sum(r["stop_reason"] == "max_tokens" for r in rows),
        "output_tokens_avg": sum(r["usage"]["output"] for r in rows) / n,
        "output_tokens_max": max(r["usage"]["output"] for r in rows),
        "rules": dict(rules),
    }


def blind_order(ids: list[str], seed: int, pair: tuple[str, str] = ("v7", "v8")) -> dict[str, str]:
    """카드마다 pair 중 무엇이 A인지 정한다. 절반씩 나오도록 섞는다."""
    flags = [False] * (len(ids) // 2) + [True] * (len(ids) - len(ids) // 2)
    random.Random(f"{seed}-blind").shuffle(flags)
    return {
        pid: (pair[1] if flip else pair[0]) for pid, flip in zip(sorted(ids), flags, strict=True)
    }


def build_review_data(
    inputs: list[dict], rows: list[dict], seed: int, pair: tuple[str, str] = ("v7", "v8")
) -> tuple[list[dict], dict[str, str]]:
    by_key = {(r["id"], r["version"]): r for r in rows}
    ready = [i for i in inputs if all((i["id"], v) in by_key for v in pair)]
    a_is = blind_order([i["id"] for i in ready], seed, pair)
    items = []
    for row in ready:
        first = a_is[row["id"]]
        second = pair[1] if first == pair[0] else pair[0]
        prompt = row["prompt"]
        if len(prompt) > PROMPT_PREVIEW_CHARS:
            prompt = prompt[:PROMPT_PREVIEW_CHARS] + "\n…(이하 생략)"
        items.append(
            {
                "id": row["id"],
                "input": prompt,
                "a": by_key[(row["id"], first)]["text"],
                "b": by_key[(row["id"], second)]["text"],
            }
        )
    return items, a_is


def with_final_period(text: str, stop_reason: str) -> str:
    """정상 종료된 글의 마지막에 문장부호가 없으면 마침표를 붙인다.

    Gemma가 "~어요"로 끝낸 뒤 마침표를 빼는 일이 30건 중 7건 있었다. 가려진 비교에서 마침표 유무가
    모델을 드러내지 않도록 양쪽에 같은 규칙을 적용한다. 글이 max_tokens에서 잘린 경우는 건드리지
    않는다(잘린 글에 마침표를 붙이면 정상 글처럼 보인다).
    """
    text = text.strip()
    if not text or stop_reason == "max_tokens" or re.search(r"[.!?…]$", text):
        return text
    return text + "."


def relabel_rows(rows: list[dict], label: str, version: str, ids: set[str]) -> list[dict]:
    """한 모델의 결과 중 `version`만 골라, 비교 화면이 "버전"으로 구분할 수 있게 이름을 바꾼다."""
    name = f"{label} {version}"
    return [
        {**r, "version": name, "text": with_final_period(r["text"], r["stop_reason"])}
        for r in rows
        if r["id"] in ids and r["version"] == version
    ]


def compare_models(args: argparse.Namespace) -> None:
    inputs = make_inputs(args.count, args.seed)
    ids = {i["id"] for i in inputs}
    rows: list[dict] = []
    for label in (args.a, args.b):
        found = relabel_rows(read_jsonl(output_path(label)), label, args.version, ids)
        if not found:
            sys.exit(f"{label}의 {args.version} 결과가 없어요. 먼저 run을 실행하세요.")
        rows += found
    pair = (f"{args.a} {args.version}", f"{args.b} {args.version}")
    items, a_is = build_review_data(inputs, rows, args.seed, pair)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    note = (
        "같은 프롬프트로 만든 두 모델의 글입니다. 마침표가 빠진 글은 양쪽 모두 마침표를 붙여 "
        "보여 드립니다(어느 쪽이 어느 모델인지 마침표로 드러나지 않게 하려는 것)."
    )
    data = {"items": items, "a_is": a_is, "pair": list(pair), "note": note}
    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, ensure_ascii=False)
    )
    name = f"review_models_{args.a}_{args.b}_{args.version}.html".replace("/", "_")
    page_path = OUT_DIR / name
    page_path.write_text(page, encoding="utf-8")
    print(f"비교 화면: {page_path} ({len(items)}건)")


def report(args: argparse.Namespace) -> None:
    inputs = make_inputs(args.count, args.seed)
    prompts = {i["id"]: i["prompt"] for i in inputs}
    rows = [r for r in read_jsonl(output_path(args.label or args.model)) if r["id"] in prompts]
    if not rows:
        sys.exit("저장된 결과가 없어요. 먼저 run을 실행하세요.")
    # 채점 기준이 늘어도 예전 결과에 같은 기준이 적용되도록, 저장된 위반 목록 대신 다시 채점한다.
    for r in rows:
        r["violations"] = violations_for(r["version"], r["text"], prompts[r["id"]], r.get("ending"))
        if r["stop_reason"] == "max_tokens":
            r["violations"].append("max_tokens에서 잘림")
    print(f"모델 {args.model} | 결과 파일 {args.label or args.model} | 결과 {len(rows)}건\n")
    versions = [v for v in VERSIONS if any(r["version"] == v for r in rows)]
    summary = {v: metrics([r for r in rows if r["version"] == v]) for v in versions}
    labels = [
        ("건수", "n", "{:d}"),
        ("평균 글자 수", "chars_avg", "{:.0f}"),
        ("최대 글자 수", "chars_max", "{:d}"),
        ("평균 문장 수", "sentences_avg", "{:.1f}"),
        ("문장당 쉼표 수(나열 정도)", "commas_per_sentence", "{:.2f}"),
        ("시작 표현 다양성(앞 12글자 종류/건수)", "opening_variety", "{:.2f}"),
        ("마지막 문장 끝 7글자 종류/건수", "ending_variety", "{:.2f}"),
        ("가장 흔한 끝맺음이 차지하는 비율", "top_ending_ratio", "{:.0%}"),
        ("지정한 끝맺음을 따른 글", "ending_followed", "{:d}"),
        ("가장 많은 글에 공통인 6글자 표현의 비율(문장 안)", "shared_ratio", "{:.0%}"),
        ("규칙 위반이 있는 글", "with_violation", "{:d}"),
        ("max_tokens에서 잘린 글", "truncated", "{:d}"),
        ("평균 출력 토큰", "output_tokens_avg", "{:.0f}"),
        ("최대 출력 토큰", "output_tokens_max", "{:d}"),
    ]
    print(f"{'항목':<40}" + "".join(f"{v:>10}" for v in summary))
    for label, key, fmt in labels:
        cells = ["-" if s[key] is None else fmt.format(s[key]) for s in summary.values()]
        print(f"{label:<40}" + "".join(f"{cell:>10}" for cell in cells))
    for version, s in summary.items():
        print(f"\n{version} 가장 많이 겹친 표현: '{s['shared_phrase']}'")
        if s["rules"]:
            print(f"\n{version} 위반 내용: {s['rules']}")

    pair = tuple(args.pair)
    items, a_is = build_review_data(inputs, rows, args.seed, pair)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {"items": items, "a_is": a_is, "pair": list(pair)}
    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, ensure_ascii=False)
    )
    page_path = OUT_DIR / f"review_{pair[0]}_{pair[1]}.html"
    page_path.write_text(page, encoding="utf-8")
    print(f"\n비교 화면: {page_path} ({len(items)}건)")


def main() -> None:
    parser = argparse.ArgumentParser(description="RECAP 요약 프롬프트 버전 비교")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "report"):
        p = sub.add_parser(name)
        # Claude 모델 이름 또는(--backend vllm일 때) 서버의 서빙 이름. 결과 파일 이름에도 쓴다.
        p.add_argument("--model", default="claude-haiku-5-5")
        # 같은 서빙 이름에 가중치만 바뀌는 경우(FP8→AWQ 등) 결과가 섞이지 않게 파일을 따로 정한다.
        p.add_argument("--label", default="", help="결과 파일 이름(기본: --model)")
        p.add_argument("--count", type=int, default=30)
        p.add_argument("--seed", type=int, default=20261108)
    versions = sorted(VERSIONS)
    sub.choices["run"].add_argument("--versions", nargs="+", choices=versions, default=versions)
    sub.choices["report"].add_argument("--pair", nargs=2, choices=versions, default=["v8", "v9"])
    sub.choices["run"].add_argument("--budget-usd", type=float, default=0.10)
    sub.choices["run"].add_argument("--dry-run", action="store_true")
    sub.choices["run"].add_argument("--backend", choices=["anthropic", "vllm"], default="anthropic")
    sub.choices["run"].add_argument("--base-url", default="", help="vLLM 주소 (끝은 /v1)")
    models = sub.add_parser("compare-models", help="같은 버전을 두 모델이 쓴 글을 가려서 비교")
    models.add_argument(
        "--a", required=True, help="한쪽 결과 파일 이름(run의 --label 또는 --model)"
    )
    models.add_argument("--b", required=True, help="다른 쪽 결과 파일 이름")
    models.add_argument("--version", choices=versions, default="v9")
    models.add_argument("--count", type=int, default=30)
    models.add_argument("--seed", type=int, default=20261108)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "compare-models":
        compare_models(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
