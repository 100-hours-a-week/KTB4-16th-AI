"""기능6 오늘의 질문 생성 평가 — 고정 재료로 LLM(Claude·로컬)을 바꿔 가며 비교한다.

재료: evals/balance_questions/stats.json (그룹 6개, 실제 DB 대신 고정 집계)
그룹마다 질문 COUNT개를 만들어서

자동으로 재는 것
  - 형식 성공률: 요청한 개수 중 "질문 | A | B"로 읽힌 비율
  - 길이 위반: 질문 50자·선택지 20자 초과
  - 지어낸 가수: 곡 풀에 있는 가수 이름이 질문에 나오는데 그 그룹 재료에는 없는 경우
  - 중복: 같은 그룹 안에서 글자가 70% 이상 겹치는 질문 쌍
  - 금지 주제 단어 (정치·종교·외모 등)
사람이 판정할 것 (결과 CSV 빈칸)
  - 반반 갈릴까 (1~3), 자연스러운가 (1~3)

실행 (서버 설정 그대로 — LLM_KNOWLEDGE_PROVIDER=local이면 로컬 LLM):
  python scripts/eval_balance_questions.py [결과 CSV 경로] [--count 5]
"""

import argparse
import asyncio
import csv
import json
import time
from difflib import SequenceMatcher
from pathlib import Path

from app.components.question_generator import (
    OPTION_MAX_LEN,
    PROMPT_VERSION,
    QUESTION_MAX_LEN,
    QuestionGenerator,
)
from app.config import get_settings
from app.db.repositories.balance_stats_repository import GenreStats
from app.dependencies import get_clients

ROOT = Path(__file__).resolve().parents[1]
STATS = ROOT / "evals" / "balance_questions" / "stats.json"
BANNED = ["정치", "대통령", "종교", "교회", "외모", "얼굴", "몸매", "성적", "섹시", "혐오"]
DUP_RATIO = 0.7


def load_stats() -> dict[str, GenreStats]:
    groups = json.loads(STATS.read_text(encoding="utf-8"))["groups"]
    return {
        name: GenreStats(
            record_count=g["record_count"],
            top_artists=[tuple(x) for x in g["top_artists"]],
            top_moods=[tuple(x) for x in g["top_moods"]],
            top_places=[tuple(x) for x in g["top_places"]],
            top_weathers=[tuple(x) for x in g["top_weathers"]],
        )
        for name, g in groups.items()
    }


def pool_artists() -> set[str]:
    names = set()
    pool = Path(get_settings().song_pool_path)
    if not pool.exists():
        return names
    for s in json.loads(pool.read_text(encoding="utf-8"))["songs"]:
        base = s["artist"].split("(")[0].strip()
        if len(base) >= 2:
            names.add(base)
    return names


async def run(out_path: Path, count: int) -> None:
    generator = QuestionGenerator(get_clients().llm_knowledge)
    known_artists = pool_artists()
    rows, requested, timings = [], 0, []

    for group, stats in load_stats().items():
        requested += count
        started = time.perf_counter()
        try:
            questions = await generator.generate(group, stats, count)
        except Exception as e:
            print(f"{group}: 실패 {type(e).__name__}")
            questions = []
        timings.append(time.perf_counter() - started)
        allowed = {a for a, _ in stats.top_artists}
        for q, a, b in questions:
            text = f"{q} {a} {b}"
            invented = sorted(n for n in known_artists if n in text and n not in allowed)
            rows.append(
                {
                    "group": group,
                    "question": q,
                    "option_a": a,
                    "option_b": b,
                    "too_long": int(
                        len(q) > QUESTION_MAX_LEN or max(len(a), len(b)) > OPTION_MAX_LEN
                    ),
                    "invented_artist": "|".join(invented),
                    "banned_word": "|".join(w for w in BANNED if w in text),
                    "split_50_50 (1~3)": "",
                    "natural (1~3)": "",
                }
            )
        print(f"{group}: {len(questions)}/{count}개")

    dup_pairs = 0
    for group in {r["group"] for r in rows}:
        qs = [r["question"] for r in rows if r["group"] == group]
        dup_pairs += sum(
            SequenceMatcher(None, x, y).ratio() >= DUP_RATIO
            for i, x in enumerate(qs)
            for y in qs[i + 1 :]
        )

    if rows:
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n = len(rows)
    print(f"\n모델 {get_clients().llm_knowledge.model}, 프롬프트 {PROMPT_VERSION}")
    print(f"형식 성공 {n}/{requested}, 길이 위반 {sum(r['too_long'] for r in rows)}/{n}")
    print(
        f"지어낸 가수 {sum(bool(r['invented_artist']) for r in rows)}/{n}, "
        f"금지 단어 {sum(bool(r['banned_word']) for r in rows)}/{n}, 중복 쌍 {dup_pairs}"
    )
    print(f"그룹당 생성 시간 평균 {sum(timings) / len(timings):.1f}s")
    print(f"사람 판정: {out_path}의 split_50_50·natural 칸 (1=아님, 2=보통, 3=좋음)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default="eval_balance_questions_result.csv")
    parser.add_argument("--count", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(run(Path(args.out), args.count))
