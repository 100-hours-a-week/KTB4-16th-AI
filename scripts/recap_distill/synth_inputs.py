"""RECAP 지식 증류용 입력(가짜 월간 통계·기록)을 합성한다.

실제 서비스의 프롬프트와 똑같은 형식이 되도록, 서비스 코드를 그대로 불러서 만든다.
- 자물쇠 한 줄: app.workers.report_worker._build_record_lines
- 사진 비중·특징 집계: _category_distribution / _axis_stats
- 프롬프트 조립: ReportSummarizer.summarize (LLM 대신 입력만 가로채는 가짜 LLM을 꽂는다)

사용자(사람)마다 "자주 나오는 날씨·장소·곡 분위기"를 정해 두고, 기록의 60% 안팎을 그쪽으로
몰아서 반복 패턴이 생기게 한다. 나머지는 무작위라서 뚜렷한 패턴이 없는 달도 섞인다.

실행: PYTHONPATH=. .venv/bin/python scripts/recap_distill/synth_inputs.py \
        --count 3000 --seed 20261016
출력: data/recap_distill/inputs.jsonl (한 줄에 한 건: id, prompt, meta)
"""

import argparse
import asyncio
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

from app.components.axis_vectors import load_axis_specs
from app.components.category_vectors import load_tag_labels
from app.components.report_summarizer import ReportSummarizer
from app.db.repositories.report_repository import RecordSummary, ReportStats
from app.workers.report_worker import (
    OTHER_CATEGORY,
    _axis_stats,
    _build_record_lines,
    _category_distribution,
)
from scripts.recap_distill import pools

ROOT = Path(__file__).resolve().parents[2]
SCENE_TAGS = [ko for _clip, ko in load_tag_labels(ROOT / "app/mood_tags.yaml")]
AXES = {spec.key: spec for spec in load_axis_specs(ROOT / "app/axis_tags.yaml")}
OUT_DEFAULT = ROOT / "data/recap_distill/inputs.jsonl"


class _CapturingLLM:
    """summarize()가 LLM에 보내는 프롬프트만 받아 둔다. 호출은 실제로 하지 않는다."""

    model = "capture"

    def __init__(self) -> None:
        self.prompt = ""

    async def complete(self, *, system: str, prompt: str, max_tokens: int = 1024) -> str:
        self.prompt = prompt
        return "당신은 가짜 문장이에요."


def _weighted(rng: random.Random, weights: dict[str, int]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()))[0]


def _record_count(rng: random.Random) -> int:
    bucket = rng.choices(["few", "some", "many"], weights=[30, 40, 30])[0]
    return {"few": rng.randint(1, 5), "some": rng.randint(6, 15), "many": rng.randint(16, 35)}[
        bucket
    ]


def _ro(word: str) -> str:
    """받침이 있으면(ㄹ 받침 제외) '으로', 없으면 '로'를 붙인 말."""
    last = word[-1]
    if not ("가" <= last <= "힣"):
        return word + "로"
    final = (ord(last) - ord("가")) % 28  # 0이면 받침 없음, 8은 ㄹ 받침
    return word + ("로" if final in (0, 8) else "으로")


def _mood_text(rng: random.Random, style: str) -> str:
    s = pools.MOOD_STYLES[style]
    return (
        f"{rng.choice(s['texture'])} {_ro(rng.choice(s['genre']))}, "
        f"{rng.choice(s['feeling'])} 분위기가 {rng.choice(s['when'])}에 어울린다."
    )


def _persona(rng: random.Random, season_weather: list[str]) -> dict:
    return {
        "style": rng.choice(list(pools.MOOD_STYLES)),
        "weather": rng.sample(season_weather, k=rng.randint(1, 2)),
        "places": rng.sample(pools.PLACES, k=rng.randint(1, 2)),
        "artists": rng.sample(pools.ARTISTS, k=rng.randint(1, 3)),
        "scenes": rng.sample(SCENE_TAGS, k=rng.randint(1, 2)),
        "mood_bias": rng.randint(-15, 15),
    }


def _axes_for_record(rng: random.Random) -> dict[str, str]:
    """사진 1장의 다축 태그. 1단계(사진 종류)에 따라 2단계 축이 달라지고, 확신 없는 축은 빠진다."""
    kind_spec = AXES["kind"]
    kind = rng.choices([t.ko for t in kind_spec.tags], weights=[90, 10])[0]
    tags = {kind_spec.ko: kind}
    for spec in AXES.values():
        if spec.key == "kind" or spec.when != kind:
            continue
        if rng.random() < 0.85:
            tags[spec.ko] = rng.choice([t.ko for t in spec.tags])
    return tags


def build_one(index: int, seed: int) -> dict:
    rng = random.Random(f"{seed}-{index}")
    month = rng.randint(1, 12)
    season_weights = pools.WEATHER_BY_SEASON[pools.season_of(month)]
    persona = _persona(rng, list(season_weights))

    records: list[RecordSummary] = []
    moods: dict[int, str] = {}
    scenes: dict[int, list[str]] = {}
    axes: dict[int, dict[str, str]] = {}
    for rid in range(1, _record_count(rng) + 1):
        weather = (
            rng.choice(persona["weather"]) if rng.random() < 0.6 else _weighted(rng, season_weights)
        )
        temperature = round(pools.MONTH_TEMPERATURE[month - 1] + rng.gauss(0, 3), 1)
        if weather in ("SNOW", "RAIN_SNOW"):
            temperature = min(temperature, 1.5)
        score = persona["mood_bias"] + pools.WEATHER_MOOD_SHIFT[weather] + rng.gauss(0, 15)
        style = persona["style"] if rng.random() < 0.7 else rng.choice(list(pools.MOOD_STYLES))
        records.append(
            RecordSummary(
                record_id=rid,
                external_track_id=f"syn-{index}-{rid}",
                created_at=datetime(2026, month, rng.randint(1, 28)),
                mood_score=None if rng.random() < 0.05 else int(max(-50, min(50, score))),
                weather_condition=None if rng.random() < 0.03 else weather,
                temperature=None if rng.random() < 0.03 else temperature,
                comment=rng.choice(pools.COMMENTS) if rng.random() < 0.55 else None,
                artist_name=(
                    rng.choice(persona["artists"])
                    if rng.random() < 0.5
                    else rng.choice(pools.ARTISTS)
                ),
                place_name=(
                    None
                    if rng.random() < 0.05
                    else rng.choice(persona["places"])
                    if rng.random() < 0.6
                    else rng.choice(pools.PLACES)
                ),
            )
        )
        if rng.random() > 0.05:  # 5%는 아직 임베딩 전이라 곡 분위기 글이 없다
            moods[rid] = _mood_text(rng, style)
        if rng.random() < 0.9:  # 10%는 장소 태그가 임계치에 못 미친다 → 비중에서 "기타"
            scenes[rid] = [
                rng.choice(persona["scenes"]) if rng.random() < 0.5 else rng.choice(SCENE_TAGS)
            ]
        else:
            scenes[rid] = []
        axes[rid] = _axes_for_record(rng)

    artists = Counter(r.artist_name for r in records if r.artist_name)
    places = Counter(r.place_name for r in records if r.place_name)
    scores = [r.mood_score for r in records if r.mood_score is not None]
    stats = ReportStats(
        avg_mood_score=sum(scores) / len(scores) if scores else None,
        top_artist=artists.most_common(1)[0][0] if artists else None,
        top_place=places.most_common(1)[0][0] if places else None,
    )

    capture = _CapturingLLM()
    asyncio.run(
        ReportSummarizer(capture).summarize(
            _build_record_lines(records, moods, scenes),
            stats,
            _category_distribution(scenes),
            _axis_stats(axes),
        )
    )
    return {
        "id": f"syn-{index:05d}",
        "prompt": capture.prompt,
        "meta": {
            "month": month,
            "records": len(records),
            "persona": {k: persona[k] for k in ("style", "weather", "places", "artists", "scenes")},
            "other_category": OTHER_CATEGORY,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RECAP 증류용 가짜 입력 합성")
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i in range(args.count):
            f.write(json.dumps(build_one(i, args.seed), ensure_ascii=False) + "\n")
    print(f"{args.count}건을 {args.out}에 저장했어요.")


if __name__ == "__main__":
    main()
