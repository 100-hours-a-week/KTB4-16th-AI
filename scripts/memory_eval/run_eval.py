"""기능5 기억 검색 정확도 측정 — 질문 30개로 "상위 3개 안에 정답이 있는 비율(Hit@3)"을 잰다.

서비스 코드(질문 해석·조건 점수·점수 합치기)는 실제 그대로 돌리고, DB 두 곳(pgvector 검색, 백엔드
MySQL 상세 조회)만 메모리 가짜로 바꾼다. 임베딩은 실제 KURE-v1 ONNX를 쓴다. 외부 호출·비용은 없다.
그래서 실제 SQL(pgvector·MySQL 쿼리)은 이 측정에 포함되지 않는다.

지표
- Hit@3: 질문마다 결과 위 3개 안에 정답 기록이 하나라도 있으면 성공. 성공한 질문 수 / 전체
- Precision@3: 위 3개 중 정답 비율의 평균 (정답이 3건보다 적으면 만점이 못 된다)
- MRR: 첫 정답의 순위 역수의 평균 (결과 전체 기준)
- 정답 없는 질문: 결과가 비어 있으면 성공 (Hit@3에는 넣지 않고 따로 센다)
- 무작위 기준: 25건 중 정답이 g건일 때 아무 3건이나 골라도 맞을 확률.
  점수가 이 기준보다 얼마나 나은지를 봐야 한다.

실행 (PyTorch 없이 onnxruntime·tokenizers가 있는 파이썬이 필요하다):
    EXTRA_SITE_PACKAGES=<프로젝트 .venv의 site-packages> PYTHONPATH=. \\
        <kure_venv>/bin/python scripts/memory_eval/run_eval.py --model-dir <KURE ONNX 폴더>
결과: data/memory_eval/results_<이름>.md / .json
"""

import argparse
import asyncio
import json
import math
import os
import sys
from datetime import date, datetime
from pathlib import Path

if extra := os.environ.get("EXTRA_SITE_PACKAGES"):
    sys.path.append(extra)

from app.components.condition_scoring import ConditionWeights  # noqa: E402
from app.components.question_parser import QuestionParser  # noqa: E402
from app.components.vector_search import SimilarRecord  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db.repositories.memory_record_repository import ConditionRow  # noqa: E402
from app.schemas.memory_search import MemorySearchRequest  # noqa: E402
from app.services.memory_search_service import MemorySearchService  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/memory_eval"
TOP_K = 3


# ---- 가짜 저장소 ----


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))  # 벡터는 길이 1로 정규화돼 있다


class InMemoryVectorSearch:
    """pgvector 대역 — 사용자별 코멘트·곡 무드 벡터를 파이썬으로 비교한다."""

    def __init__(self, records_by_user: dict[int, list[dict]], vectors: dict[str, list[float]]):
        self._recs = records_by_user
        self._vec = vectors

    def _hits(self, user_id: int, key: str, query: list[float], ids: set[int] | None = None):
        hits = []
        for rec in self._recs.get(user_id, []):
            if ids is not None and rec["record_id"] not in ids:
                continue
            text = rec[key]
            if not text:
                continue
            hits.append(
                SimilarRecord(
                    rec["record_id"], rec["external_track_id"], _cosine(query, self._vec[text])
                )
            )
        return sorted(hits, key=lambda h: -h.similarity)

    async def similar_by_comment(self, user_id, query_vector, limit):
        return self._hits(user_id, "comment", query_vector)[:limit]

    async def similar_by_music(self, user_id, query_vector, limit):
        return self._hits(user_id, "music_mood_text", query_vector)[:limit]

    async def similarities_for(self, user_id, record_ids, query_vector):
        ids = set(record_ids)
        return (
            self._hits(user_id, "comment", query_vector, ids),
            self._hits(user_id, "music_mood_text", query_vector, ids),
        )

    async def moods_by_record(self, user_id):
        return {r["record_id"]: tuple(r["music_moods"]) for r in self._recs.get(user_id, [])}

    async def genres_by_record(self, user_id):
        return {}  # 측정 데이터에는 곡 장르가 없다


class InMemoryRecords:
    """백엔드 MySQL 대역."""

    def __init__(self, records_by_user: dict[int, list[dict]]):
        self._recs = records_by_user

    async def existing_ids(self, *, user_id, record_ids):
        wanted = set(record_ids)
        return {r["record_id"] for r in self._recs.get(user_id, []) if r["record_id"] in wanted}

    async def list_track_names(self, *, user_id, limit):
        return []  # 측정 데이터에는 곡명·가수명이 없다 (가수·곡명 조건은 측정 대상이 아니다)

    async def list_condition_rows(self, *, user_id, limit):
        rows = [
            ConditionRow(
                record_id=r["record_id"],
                created_at=datetime.fromisoformat(r["created_at"]),
                weather_condition=r["weather"],
                temperature=r["temperature"],
                mood_score=r["mood_score"],
                dong_name=r["dong_name"],
                dong_code=r["dong_code"],
                latitude=r["lat"],
                longitude=r["lng"],
                comment=r["comment"][:80],
            )
            for r in self._recs.get(user_id, [])
        ]
        return sorted(rows, key=lambda r: (r.created_at, r.record_id), reverse=True)[:limit]


class CachedEmbedder:
    """텍스트 → 벡터. 한 번 계산한 값은 파일에 저장해 두고 다시 쓴다 (모델은 처음에만 필요)."""

    model = "kure-cache"

    def __init__(self, cache_path: Path, model_dir: str | None):
        self._path = cache_path
        self._cache: dict[str, list[float]] = (
            json.loads(cache_path.read_text("utf-8")) if cache_path.exists() else {}
        )
        self._model_dir = model_dir
        self._model = None

    def warm(self, texts: list[str]) -> None:
        missing = sorted({t for t in texts if t not in self._cache})
        if not missing:
            return
        if self._model is None:
            from app.components.kure_model import load_kure_model

            s = get_settings()
            self._model = load_kure_model(
                self._model_dir or s.kure_model_dir,
                s.kure_model_name,
                s.kure_embedding_dim,
                s.kure_threads,
                s.kure_max_tokens,
            )
            if self._model is None:
                sys.exit("KURE 모델 파일을 못 찾았어요. --model-dir로 ONNX 폴더를 알려 주세요.")
        for text, vec in zip(missing, self._model.embed(missing), strict=True):
            self._cache[text] = [round(x, 6) for x in vec]
        self._path.write_text(json.dumps(self._cache, ensure_ascii=False), encoding="utf-8")

    @property
    def vectors(self) -> dict[str, list[float]]:
        return self._cache

    async def embed(self, text: str) -> list[float]:
        self.warm([text])
        return self._cache[text]


# ---- 지표 ----


def random_hit_probability(total: int, gold: int, k: int = TOP_K) -> float:
    """아무 k건이나 골랐을 때 정답이 하나라도 들어 있을 확률."""
    if gold <= 0:
        return 0.0
    return 1.0 - math.comb(total - gold, k) / math.comb(total, k)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


async def run(args) -> dict:
    dataset = json.loads((DATA / "dataset.json").read_text("utf-8"))
    users = {int(u): recs for u, recs in dataset["users"].items()}
    today = date.fromisoformat(dataset["today"])

    embedder = CachedEmbedder(DATA / "vectors_kure.json", args.model_dir)
    embedder.warm(
        [r["comment"] for recs in users.values() for r in recs]
        + [r["music_mood_text"] for recs in users.values() for r in recs]
        + [q["text"] for q in dataset["questions"]]
    )

    s = get_settings()
    parser = QuestionParser.load_default()
    service = MemorySearchService(
        embedder,
        InMemoryVectorSearch(users, embedder.vectors),
        InMemoryRecords(users),
        min_similarity=args.min_similarity
        if args.min_similarity is not None
        else s.memory_min_similarity,
        comment_weight=s.memory_search_comment_weight,
        music_weight=s.memory_search_music_weight,
        candidates=s.memory_search_candidates,
        parser=parser,
        condition_weight=(
            args.condition_weight
            if args.condition_weight is not None
            else s.memory_search_condition_weight
        ),
        condition_weights=ConditionWeights(
            location=s.memory_search_location_weight, other=s.memory_search_other_condition_weight
        ),
        scan_limit=s.memory_search_scan_limit,
        condition_candidates=s.memory_search_condition_candidates,
        display_count=s.memory_search_display_count,
        suggestion_count=s.memory_search_suggestion_count,
        today=lambda: today,
    )

    results = []
    for q in dataset["questions"]:
        res = await service.search(
            MemorySearchRequest(user_id=q["user"], query_text=q["text"], limit=50)
        )
        returned = [r.record_id for r in res.records]
        gold = set(q["gold"])
        top = returned[:TOP_K]
        n_user = len(users[q["user"]])
        first = next((i + 1 for i, rid in enumerate(returned) if rid in gold), None)
        results.append(
            {
                "id": q["id"],
                "category": q["category"],
                "text": q["text"],
                "user": q["user"],
                "expect_none": bool(q.get("expect_none")),
                "gold": sorted(gold),
                "top3": top,
                "returned": len(returned),
                "suggested": len(res.suggested_records),
                "understood": [c.label for c in parser.parse(q["text"], today)],
                "answer": res.answer.splitlines()[0] if res.answer else "",
                "hit": bool(gold & set(top)) if gold else None,
                "precision": len(gold & set(top)) / TOP_K if gold else None,
                "rr": (1.0 / first) if first else 0.0,
                "empty_ok": (len(returned) == 0) if q.get("expect_none") else None,
                "random_hit": random_hit_probability(n_user, len(gold)),
            }
        )
    return {
        "settings": {
            "min_similarity": service._min_similarity,
            "condition_weight": service._condition_weight,
            "today": today.isoformat(),
        },
        "results": results,
    }


def render(out: dict) -> str:
    results = out["results"]
    scored = [r for r in results if not r["expect_none"]]
    none = [r for r in results if r["expect_none"]]
    hits = sum(r["hit"] for r in scored)
    low, high = wilson(hits, len(scored))
    lines = [
        "# 기능5 기억 검색 정확도 측정 결과",
        "",
        f"- 설정: 최소 유사도 {out['settings']['min_similarity']}, 조건 가중치 "
        f"{out['settings']['condition_weight']}, 기준일 {out['settings']['today']}",
        f"- **Hit@3 {hits}/{len(scored)} = {hits / len(scored):.0%}** "
        f"(95% 신뢰구간 {low:.0%}~{high:.0%}) — 정답이 있는 질문만",
        f"- 무작위로 골랐을 때의 평균 성공 확률(기준): "
        f"{sum(r['random_hit'] for r in scored) / len(scored):.0%}",
        f"- Precision@3 평균 {sum(r['precision'] for r in scored) / len(scored):.2f}, "
        f"MRR 평균 {sum(r['rr'] for r in scored) / len(scored):.2f}",
        f"- 정답 없는 질문: {sum(r['empty_ok'] for r in none)}/{len(none)}개에서 결과가 비어 있음",
        "",
        "## 유형별",
        "",
        "| 유형 | 질문 수 | Hit@3 | 무작위 기준 |",
        "|---|---|---|---|",
    ]
    for cat in dict.fromkeys(r["category"] for r in scored):
        rows = [r for r in scored if r["category"] == cat]
        lines.append(
            f"| {cat} | {len(rows)} | {sum(r['hit'] for r in rows)}/{len(rows)} | "
            f"{sum(r['random_hit'] for r in rows) / len(rows):.0%} |"
        )
    lines += [
        "",
        "## 질문별",
        "",
        "| 번호 | 질문 | 해석한 조건 | 정답 | 상위 3 | 결과 수 | 성공 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ok = ("○" if r["empty_ok"] else "✗") if r["expect_none"] else ("○" if r["hit"] else "✗")
        lines.append(
            f"| {r['id']} | {r['text']} | {', '.join(r['understood']) or '(없음)'} | "
            f"{r['gold'] or '없음'} | {r['top3'] or '비어 있음'} | {r['returned']} | {ok} |"
        )
    wrong = [r for r in results if (r["empty_ok"] is False) or (r["hit"] is False)]
    lines += ["", f"## 틀린 질문 ({len(wrong)}개)", ""]
    for r in wrong:
        lines += [
            f"- **{r['id']} {r['text']}** — 해석: {', '.join(r['understood']) or '(조건 없음)'} | "
            f"정답 {r['gold'] or '없음'} | 결과 {r['returned']}건, 상위 {r['top3'] or '없음'} | "
            f"안내: {r['answer']}"
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="기능5 기억 검색 정확도 측정")
    parser.add_argument("--model-dir", default=os.environ.get("KURE_MODEL_DIR"))
    parser.add_argument("--min-similarity", type=float, default=None)
    parser.add_argument("--condition-weight", type=float, default=None)
    parser.add_argument("--name", default="baseline", help="결과 파일 이름 (results_<이름>.md)")
    args = parser.parse_args()

    out = asyncio.run(run(args))
    (DATA / f"results_{args.name}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    text = render(out)
    (DATA / f"results_{args.name}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
