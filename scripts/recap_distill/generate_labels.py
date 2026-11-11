"""가짜 입력(inputs.jsonl)에 Claude로 "정답 회고 문장"을 붙인다 (지식 증류의 정답 만들기).

고정해 둔 v7 시스템 프롬프트(prompt_v7.py)를 쓰고, 결과마다 자동 채점(rubric.py)을 붙여
저장한다. 외부 API(Claude)를 부르는 일이라 비용이 든다. 그래서 아래 안전장치를 둔다.
- --budget-usd: 지금까지 쓴 비용이 이 값에 가까워지면 다음 호출 전에 멈춘다 (실제 토큰 수로 계산).
- --dry-run: API를 부르지 않고 건수·예상 비용만 보여 준다.
- 이어하기: 출력 파일에 이미 있는 id는 건너뛴다. 멈췄다가 다시 돌려도 중복 비용이 없다.
- API 키는 앱 설정(.env의 ANTHROPIC_API_KEY)에서 읽고, 화면에 출력하지 않는다.
입력은 전부 가짜 데이터다(synth_inputs.py). 실제 사용자 기록은 보내지 않는다.

실행 예 (30건 시험):
    PYTHONPATH=. .venv/bin/python scripts/recap_distill/generate_labels.py \\
        --limit 30 --budget-usd 0.10
"""

import argparse
import json
import sys
from pathlib import Path

import anthropic

from app.config import get_settings
from scripts.recap_distill.prompt_v7 import V7_SYSTEM_PROMPT as SYSTEM_PROMPT
from scripts.recap_distill.rubric import check

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/recap_distill"

# 달러 / 100만 토큰 (입력, 출력). 공식 가격표(platform.claude.com/docs/en/about-claude/pricing)에서
# 2026-10-16에 확인한 값. 모델을 추가하면 가격도 같이 적을 것 — 모르는 모델은 실행하지 않는다.
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}
MAX_OUTPUT_TOKENS = 150  # v7 서비스가 쓰던 값(서비스는 v9로 바뀌어 300)
# 한국어는 1글자가 대략 1토큰이라고 넉넉하게 잡은 사전 추정용 값 (실제 비용은 응답의 토큰 수로 계산)
EST_OUTPUT_TOKENS = 100
EST_TOKENS_PER_CHAR = 1.1  # 시험에서 입력 토큰이 글자 수의 약 1.03배였다 (Haiku 5.5, 30건)


def cost_of(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


def estimate_cost(model: str, prompts: list[str]) -> float:
    chars = [len(SYSTEM_PROMPT) + len(p) for p in prompts]
    return sum(cost_of(model, int(c * EST_TOKENS_PER_CHAR), EST_OUTPUT_TOKENS) for c in chars)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser(description="Claude로 정답 회고 문장 생성")
    parser.add_argument("--model", default="claude-haiku-5-5", choices=sorted(PRICES))
    parser.add_argument("--inputs", type=Path, default=DATA / "inputs.jsonl")
    parser.add_argument(
        "--out", type=Path, default=None, help="기본: data/recap_distill/labels_<모델>.jsonl"
    )
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--budget-usd", type=float, default=0.10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out = args.out or DATA / f"labels_{args.model}.jsonl"
    done = {row["id"] for row in read_jsonl(out)}
    todo = [r for r in read_jsonl(args.inputs)[: args.limit] if r["id"] not in done]
    est = estimate_cost(args.model, [r["prompt"] for r in todo])
    print(f"모델 {args.model} | 이번에 만들 건수 {len(todo)} (이미 있는 {len(done)}건 건너뜀)")
    print(f"사전 추정 비용(최대치 가정): ${est:.4f} | 한도 ${args.budget_usd:.2f}")
    if args.dry_run or not todo:
        return
    if est > args.budget_usd:
        print("추정 비용이 한도를 넘어서 시작하지 않아요. --limit이나 --budget-usd를 조정하세요.")
        sys.exit(1)

    api_key = get_settings().anthropic_api_key
    if not api_key:
        sys.exit(".env에 ANTHROPIC_API_KEY가 없어요.")
    client = anthropic.Anthropic(api_key=api_key, max_retries=1, timeout=60)

    spent, passed = 0.0, 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for n, row in enumerate(todo, start=1):
            # 다음 1건이 평균보다 조금 더 들어도 한도를 넘지 않을 때만 부른다
            average = spent / (n - 1) if n > 1 else est / len(todo)
            if spent + average * 1.5 > args.budget_usd:
                print(f"한도에 가까워서 {n - 1}건에서 멈춰요 (사용 ${spent:.4f}).")
                break
            try:
                res = client.messages.create(
                    model=args.model,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": row["prompt"]}],
                    # 서비스(app/clients/anthropic_client.py)와 같은 옵션. 이게 없으면 모델이
                    # "생각"에 max_tokens를 다 써서 문장이 빈 채로 끝나는 일이 잦았다
                    # (시험 30건 중 17~19건)
                    output_config={"effort": "low"},
                )
            except anthropic.APIStatusError as e:
                print(f"API 오류 (status {e.status_code}) — {n - 1}건에서 멈춰요.")
                break
            except anthropic.APIConnectionError:
                print(f"API 연결 실패 — {n - 1}건에서 멈춰요.")
                break
            text = "".join(b.text for b in res.content if b.type == "text").strip()
            used = cost_of(args.model, res.usage.input_tokens, res.usage.output_tokens)
            spent += used
            if not text:  # 저장하지 않는다 — 다시 실행하면 이 건을 다시 시도한다
                print(f"{row['id']}: 빈 응답 (정지 사유 {res.stop_reason}) — 건너뜀")
                continue
            violations = check(text, row["prompt"])
            if res.stop_reason == "max_tokens":  # 문장이 중간에 잘렸거나 군말이 섞인 것
                violations.append("max_tokens에서 잘림")
            passed += not violations
            record = {
                "id": row["id"],
                "text": text,
                "model": args.model,
                "stop_reason": res.stop_reason,
                "usage": {"input": res.usage.input_tokens, "output": res.usage.output_tokens},
                "violations": violations,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
    rows = read_jsonl(out)
    ok = sum(not r["violations"] for r in rows)
    print(f"이번 실행 비용 ${spent:.4f} | 파일에 있는 {len(rows)}건 중 자동 채점 통과 {ok}건")


if __name__ == "__main__":
    main()
