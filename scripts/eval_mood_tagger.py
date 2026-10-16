"""기능4 곡 무드 태깅 평가 — 같은 정답지로 LLM(Claude·로컬)을 바꿔 가며 비교한다.

정답지: evals/mood_tagger/apple_kr_top100.csv — 사람이 모은 데이터라 공개 레포에 두지 않는다.
  비공개 HF 저장소 dionypark/mulo-song-pool의 evals/에서 받아 이 경로에 둔다 (.gitignore)
  - melon_genres: 사람이 멜론에서 옮겨 적은 장르(여러 개는 |로)
  - baseline_*: 2026-10-15 서비스 태거(claude-sonnet-5, music-mood-v4)의 결과와 사람 무드 판정

자동으로 재는 것 (사람 손 없음)
  - 장르 엄격 일치: 태거 장르가 멜론 장르 목록에 있음 (멜론 '국내·국외 드라마·영화'는 OST로 봄)
  - 장르 비슷한 묶음 일치: 인디·발라드·포크·록 / 랩·R&B / 댄스·POP·일렉 묶음 안이면 일치
  - 형식 실패율: 장르나 무드를 목록에서 못 읽은 비율
  - 평균·p95 응답 시간
무드가 맞는지는 사람이 판정해야 해서, 결과 CSV에 무드 O/X 칸을 비워 둔다.

실행 (서버 설정을 그대로 쓴다 — LLM_KNOWLEDGE_PROVIDER=local이면 로컬 LLM):
  python scripts/eval_mood_tagger.py [결과 CSV 경로] [--limit N]
"""

import argparse
import asyncio
import csv
import time
from pathlib import Path

from app.components.music_mood_tagger import PROMPT_VERSION, MusicMoodTagger
from app.dependencies import get_clients

EVAL_SET = Path(__file__).resolve().parents[1] / "evals" / "mood_tagger" / "apple_kr_top100.csv"
NEAR_GROUPS = [
    {"인디음악", "발라드", "포크/블루스", "록/메탈"},
    {"랩/힙합", "R&B/Soul"},
    {"댄스", "POP", "일렉트로니카"},
]
OST_LIKE = ("드라마", "영화")


def strict_hit(genre: str, melon: list[str]) -> bool:
    if not genre:
        return False
    if genre == "OST" and any(k in m for m in melon for k in OST_LIKE):
        return True
    return genre in melon


def near_hit(genre: str, melon: list[str]) -> bool:
    if strict_hit(genre, melon):
        return True
    return any(genre in g and m in g for g in NEAR_GROUPS for m in melon)


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if ordered else 0.0


async def run(out_path: Path, limit: int | None) -> None:
    rows = list(csv.DictReader(EVAL_SET.open(encoding="utf-8")))[:limit]
    tagger = MusicMoodTagger(get_clients().llm_knowledge)

    results = []
    for row in rows:
        started = time.perf_counter()
        try:
            d = await tagger.describe(row["title"], row["artist"])
            genre, moods, text, error = d.genre, list(d.moods), d.text, ""
        except Exception as e:  # 한 곡 실패가 평가 전체를 멈추지 않게
            genre, moods, text, error = "", [], "", type(e).__name__
        results.append(
            {
                **row,
                "genre": genre,
                "moods": "|".join(moods),
                "text": text,
                "error": error,
                "seconds": round(time.perf_counter() - started, 2),
            }
        )
        print(f"{len(results)}/{len(rows)} {row['artist']} - {row['title']}: {genre} {moods}")

    melon = [r["melon_genres"].split("|") for r in results]
    n = len(results)
    pairs = list(zip(results, melon, strict=True))
    new_strict = sum(strict_hit(r["genre"], m) for r, m in pairs)
    new_near = sum(near_hit(r["genre"], m) for r, m in pairs)
    old_strict = sum(strict_hit(r["baseline_genre"], m) for r, m in pairs)
    old_near = sum(near_hit(r["baseline_genre"], m) for r, m in pairs)
    format_fail = sum(1 for r in results if not r["genre"] or not r["moods"])
    seconds = [r["seconds"] for r in results]

    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = [*results[0].keys(), "mood_ok (O/X)"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({**r, "mood_ok (O/X)": ""} for r in results)

    model = get_clients().llm_knowledge.model
    print(f"\n모델 {model}, 프롬프트 {PROMPT_VERSION}, {n}곡")
    print(f"장르 엄격 일치   기준선 {old_strict}/{n} → 이번 {new_strict}/{n}")
    print(f"장르 묶음 일치   기준선 {old_near}/{n} → 이번 {new_near}/{n}")
    print(f"형식 실패 {format_fail}/{n}")
    print(f"응답 시간 평균 {sum(seconds) / n:.2f}s, p95 {p95(seconds):.2f}s")
    print(f"무드는 {out_path}의 mood_ok 칸에 사람이 O/X — 기준선 무드 정확도는 100/100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default="eval_mood_tagger_result.csv")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(run(Path(args.out), args.limit))
