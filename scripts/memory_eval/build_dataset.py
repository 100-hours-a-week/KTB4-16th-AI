"""기능5 기억 검색 정확도 측정용 시험 데이터를 만든다 (가상 사용자 4명 × 자물쇠 25건).

자물쇠의 코멘트는 임베딩 평가 때 만든 100개(eval/text_embedding/corpus.jsonl)를 그대로 쓰고, 나머지
항목(장소·날씨·기온·기분·날짜·곡 무드)은 무작위로 붙인다. 질문(questions.yaml)마다
"정답 기준"이 있어서,
정답은 그 기준을 만족하는 기록으로 자동 계산한다 — 사람이 하나씩 라벨링하지 않는다.
다만 코멘트 의미 질문은 임베딩 평가에서 사람이 정한 정답을 그대로 쓴다.

질문마다 정답이 너무 적으면 무작위만으로는 측정이 안 되어서, 조건에 맞는 기록을 몇 개 "심는다"
(그 기록의 속성만 바꾼다). 어떤 기록을 심었는지 dataset.json의 planted에 남긴다.
경계 근처 값(기온 5/6도, 15/16도, 25/26도, 기분 ±20 근처, 명소 반경 근처)은 일부러 만들지 않아서
"애매한 정답"이 없게 했다.

실행: PYTHONPATH=. .venv/bin/python scripts/memory_eval/build_dataset.py
출력: data/memory_eval/dataset.json, data/memory_eval/questions_draft.md
      (questions_draft.md는 사람이 읽는 질문·정답 요약)
"""

import argparse
import json
import random
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import yaml

from app.components.geo import haversine_km
from app.components.music_tags import MOOD_VOCAB
from app.components.place_dictionary import PlaceDictionary

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = ROOT / "scripts/memory_eval/questions.yaml"
EVAL_DIR = ROOT / "eval/text_embedding"
OUT_DIR = ROOT / "data/memory_eval"
TODAY = date(2026, 10, 16)  # 측정 중 "오늘". "지난달" = 2026-09

# (법정동 이름, 법정동코드 10자리, 위도, 경도). 시도·시군구 앞자리는 regions.yaml 기준,
# 나머지 자리는 임의
LOCATIONS = {
    "성수동1가": ("1120011500", 37.5440, 127.0370),
    "성수동2가": ("1120011400", 37.5470, 127.0560),
    "연남동": ("1144012700", 37.5640, 126.9240),
    "서교동": ("1144012000", 37.5560, 126.9230),
    "망원동": ("1144012400", 37.5565, 126.9010),
    "역삼동": ("1168010100", 37.5006, 127.0364),
    "여의도동": ("1156011000", 37.5219, 126.9245),
    "잠실동": ("1171010100", 37.5133, 127.1000),
    "판교동": ("4113510300", 37.3948, 127.1112),
    "정자동": ("4113510100", 37.3669, 127.1083),
    "영통동": ("4111710300", 37.2596, 127.0467),
    "행궁동": ("4111113900", 37.2827, 127.0143),
    "오산동": ("4159010100", 37.2090, 127.0780),
}
HOME = {
    1: ["성수동1가", "성수동2가", "연남동", "역삼동", "여의도동", "서교동", "망원동", "잠실동"],
    2: ["서교동", "연남동", "망원동", "잠실동", "역삼동", "여의도동", "성수동1가", "성수동2가"],
    3: ["판교동", "정자동", "성수동1가", "역삼동", "여의도동", "잠실동", "연남동", "망원동"],
    4: ["판교동", "정자동", "역삼동", "잠실동", "오산동", "성수동1가", "연남동", "망원동"],
}
# 사용자별 기록 기간. 질문이 가리키는 기간이 일부러 비는 곳도 있다 (m-29: 사용자 2는 2025년 봄 없음)
DATE_WINDOW = {
    1: (date(2025, 9, 1), date(2026, 10, 10)),
    2: (date(2025, 6, 1), date(2026, 10, 14)),
    3: (date(2025, 9, 1), date(2026, 10, 12)),
    4: (date(2025, 3, 1), date(2026, 10, 14)),
}
SEASON_WEATHER = {
    "winter": {"CLEAR": 30, "CLOUDY": 15, "OVERCAST": 20, "SNOW": 20, "RAIN_SNOW": 8, "RAIN": 7},
    "spring": {"CLEAR": 40, "CLOUDY": 20, "OVERCAST": 12, "RAIN": 20, "SHOWER": 8},
    "summer": {"CLEAR": 30, "CLOUDY": 20, "OVERCAST": 10, "RAIN": 20, "SHOWER": 20},
    "autumn": {"CLEAR": 45, "CLOUDY": 25, "OVERCAST": 15, "RAIN": 15},
}
MONTH_TEMP = [-1, 2, 8, 14, 19, 24, 27, 28, 23, 16, 8, 1]
TEMP_BOUNDARY_ZONES = [
    (4, 7),
    (14, 17),
    (24, 27),
]  # 기온 구간 경계(5/6, 15/16, 25/26) 근처는 안 만든다
MOOD_BOUNDARY_ZONE = (17, 23)  # 기분 ±20 근처는 안 만든다
RADIUS_MARGIN_KM = 0.2  # 명소 반경 ±이 안에 있는 기록은 안 만든다
TRACKS = [
    ("밤편지", "아이유"),
    ("Love wins all", "아이유"),
    ("사건의 지평선", "윤하"),
    ("Ditto", "NewJeans"),
    ("Dynamite", "BTS"),
    ("Viva La Vida", "Coldplay"),
    ("Ocean Eyes", "Billie Eilish"),
    ("사랑하지 않아서 그랬어", "이무진"),
    ("가을 아침", "아이유"),
    ("그대에게", "장범준"),
    ("Blueming", "아이유"),
    ("Anti-Hero", "Taylor Swift"),
    ("봄날", "BTS"),
    ("Heat Waves", "Glass Animals"),
]
MOOD_TEXT = {  # 곡 무드 태그 → 곡 분위기 글 후보 (LLM이 곡마다 쓰는 1~2문장을 흉내 낸다)
    "신나는": [
        "들뜨고 신나는 리듬이 기분을 끌어올리는 곡.",
        "에너지 넘치는 비트로 신나게 달리고 싶어지는 곡.",
    ],
    "잔잔한": [
        "잔잔하게 흐르는 멜로디가 마음을 가라앉히는 곡.",
        "조용하고 잔잔한 분위기로 혼자 듣기 좋은 곡.",
    ],
    "슬픈": ["쓸쓸하고 슬픈 정서가 길게 남는 곡.", "눈물이 날 것처럼 슬픈 멜로디의 곡."],
    "몽환적인": ["꿈을 꾸는 듯 몽환적인 사운드가 번지는 곡.", "아득하고 몽환적인 분위기의 곡."],
    "따뜻한": ["포근하고 따뜻한 목소리가 감싸 주는 곡.", "따뜻한 햇살 같은 분위기의 곡."],
    "웅장한": ["웅장하게 쌓아 올리는 사운드가 벅차게 하는 곡.", "스케일이 큰 웅장한 편곡의 곡."],
    "경쾌한": ["가볍고 경쾌한 리듬이 발걸음을 가볍게 하는 곡.", "경쾌하게 통통 튀는 분위기의 곡."],
    "우울한": ["가라앉은 우울한 분위기가 이어지는 곡.", "무겁고 우울한 정서의 곡."],
    "설레는": ["두근거리는 설렘이 느껴지는 곡.", "설레는 마음을 닮은 달콤한 곡."],
    "차분한": ["차분하고 담담하게 흘러가는 곡.", "차분한 분위기로 생각을 정리하게 하는 곡."],
    "그리운": ["지나간 시간이 그리워지는 곡.", "그리움이 번지는 서정적인 곡."],
    "화려한": ["화려한 편곡과 사운드가 눈부신 곡.", "반짝이는 화려한 분위기의 곡."],
}


def season_of(month: int) -> str:
    return (
        "winter"
        if month in (12, 1, 2)
        else "spring"
        if month in (3, 4, 5)
        else "summer"
        if month in (6, 7, 8)
        else "autumn"
    )


def _in_zone(value: float, zones) -> bool:
    return any(low <= value <= high for low, high in zones)


def _random_temp(rng: random.Random, month: int, weather: str) -> float:
    while True:
        temp = round(MONTH_TEMP[month - 1] + rng.gauss(0, 3), 1)
        if weather in ("SNOW", "RAIN_SNOW"):
            temp = min(temp, 1.5)
        if not _in_zone(temp, TEMP_BOUNDARY_ZONES):
            return temp


def _random_mood(rng: random.Random) -> int:
    while True:
        score = int(max(-50, min(50, rng.gauss(0, 20))))
        if not (MOOD_BOUNDARY_ZONE[0] <= abs(score) <= MOOD_BOUNDARY_ZONE[1]):
            return score


def _random_date(rng: random.Random, window: tuple[date, date]) -> date:
    return window[0] + timedelta(days=rng.randint(0, (window[1] - window[0]).days))


def _set_place(rec: dict, name: str, lat: float | None = None, lng: float | None = None) -> None:
    code, base_lat, base_lng = LOCATIONS[name]
    rec.update(
        dong_name=name,
        dong_code=code,
        lat=base_lat if lat is None else lat,
        lng=base_lng if lng is None else lng,
    )


def _set_moods(rec: dict, tags: list[str], rng: random.Random) -> None:
    rec["music_moods"] = tags
    rec["music_mood_text"] = " ".join(rng.choice(MOOD_TEXT[t]) for t in tags)


def make_base(seed: int) -> dict[int, list[dict]]:
    rng = random.Random(seed)
    corpus = [
        json.loads(line) for line in (EVAL_DIR / "corpus.jsonl").read_text("utf-8").splitlines()
    ]
    users: dict[int, list[dict]] = defaultdict(list)
    for row in corpus:
        user = row["userId"]
        day = _random_date(rng, DATE_WINDOW[user])
        weather = rng.choices(
            list(SEASON_WEATHER[season_of(day.month)]),
            weights=list(SEASON_WEATHER[season_of(day.month)].values()),
        )[0]
        title, artist = rng.choice(TRACKS)
        rid = int(row["id"].split("-")[1])
        rec = {
            "record_id": rid,
            "user_id": user,
            "comment": row["text"],
            "created_at": f"{day.isoformat()}T18:30:00",
            "weather": weather,
            "temperature": _random_temp(rng, day.month, weather),
            "mood_score": _random_mood(rng),
            "title": title,
            "artist": artist,
            "external_track_id": f"eval-track-{rid:03d}",
        }
        _set_place(rec, rng.choice(HOME[user]))
        _set_moods(rec, rng.sample(MOOD_VOCAB, k=rng.choice([1, 1, 2])), rng)
        users[user].append(rec)
    return users


# ---- 정답 기준 ----


def _day(rec: dict) -> date:
    return date.fromisoformat(rec["created_at"][:10])


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(str(value))


def matches(rec: dict, flt: dict, places: PlaceDictionary) -> bool:
    for key, value in flt.items():
        if key == "dong_prefix" and not rec["dong_name"].startswith(value):
            return False
        if key == "region_prefix" and not rec["dong_code"].startswith(str(value)):
            return False
        if key == "near":
            entry = next(e for e in places.entries if e.name == value)
            if haversine_km(rec["lat"], rec["lng"], entry.lat, entry.lng) > entry.radius_km:
                return False
        if key == "date_range":
            start, end = _as_date(value[0]), _as_date(value[1])
            if not start <= _day(rec) <= end:
                return False
        if key == "months" and _day(rec).month not in value:
            return False
        if key == "weather" and rec["weather"] not in value:
            return False
        if key == "temp" and not value[0] <= rec["temperature"] <= value[1]:
            return False
        if key == "mood" and not value[0] <= rec["mood_score"] <= value[1]:
            return False
        if key == "song_mood" and value not in rec["music_moods"]:
            return False
    return True


def comment_gold(label: str) -> set[int]:
    for line in (EVAL_DIR / "queries.jsonl").read_text("utf-8").splitlines():
        row = json.loads(line)
        if row["id"] == label:
            return {int(cid.split("-")[1]) for cid, grade in row["gold"].items() if grade >= 1}
    raise KeyError(label)


def gold_of(question: dict, records: list[dict], places: PlaceDictionary) -> set[int]:
    flt = question["filter"]
    if question.get("expect_none") and not flt:  # 조건 없이 의미만으로 찾는 "정답 없음" 질문
        return set()
    if "comment_gold" in flt:
        return comment_gold(flt["comment_gold"])
    return {r["record_id"] for r in records if matches(r, flt, places)}


# ---- 심기 ----


def _apply(rec: dict, flt: dict, rng: random.Random, places: PlaceDictionary, window) -> None:
    """기록 한 건이 filter를 만족하도록 속성만 바꾼다."""
    if "dong_prefix" in flt:
        name = next(n for n in LOCATIONS if n.startswith(flt["dong_prefix"]))
        _set_place(rec, name)
    if "region_prefix" in flt:
        name = next(n for n, v in LOCATIONS.items() if v[0].startswith(str(flt["region_prefix"])))
        _set_place(rec, name)
    if "near" in flt:
        entry = next(e for e in places.entries if e.name == flt["near"])
        _set_place(
            rec,
            min(LOCATIONS, key=lambda n: haversine_km(*LOCATIONS[n][1:], entry.lat, entry.lng)),
            lat=round(entry.lat + rng.uniform(-0.0015, 0.0015), 4),
            lng=round(entry.lng + rng.uniform(-0.0015, 0.0015), 4),
        )
    if "date_range" in flt:
        start, end = _as_date(flt["date_range"][0]), _as_date(flt["date_range"][1])
        day = start + timedelta(days=rng.randint(0, (end - start).days))
        rec["created_at"] = f"{day.isoformat()}T18:30:00"
    if "months" in flt:
        year = rng.choice([2025, 2026]) if window[0].year == 2025 else 2026
        month = rng.choice(flt["months"])
        day = date(year, month, rng.randint(1, 28))
        day = min(max(day, window[0]), window[1])
        if day.month not in flt["months"]:  # 기간 경계로 밀려났으면 다른 해로
            day = date(day.year - 1 if day.year > 2025 else day.year + 1, month, rng.randint(1, 28))
        rec["created_at"] = f"{day.isoformat()}T18:30:00"
    if "weather" in flt:
        rec["weather"] = rng.choice(flt["weather"])
        rec["temperature"] = _random_temp(rng, _day(rec).month, rec["weather"])
    if "temp" in flt:
        low, high = flt["temp"]
        inner_low, inner_high = max(low, -10), min(high, 36)
        hot = low >= 26
        if hot:  # 더운 날은 여름 기록으로 (날짜 조건과 충돌 줄이려고 6~8월)
            rec["created_at"] = (
                f"{date(2026, rng.choice([6, 7, 8]), rng.randint(1, 28)).isoformat()}T18:30:00"
            )
            if not matches(rec, {"date_range": [window[0], window[1]]}, places):
                rec["created_at"] = (
                    f"{date(2025, rng.choice([6, 7, 8]), rng.randint(1, 28)).isoformat()}T18:30:00"
                )
        rec["temperature"] = round(rng.uniform(inner_low + 1, inner_high - 1), 1)
        rec["weather"] = (
            "SNOW" if (not hot and rec["temperature"] < 0) else rng.choice(["CLEAR", "CLOUDY"])
        )
    if "mood" in flt:
        low, high = flt["mood"]
        rec["mood_score"] = rng.randint(low + 8, high - 3) if low > -50 else rng.randint(-45, -28)
    if "song_mood" in flt:
        _set_moods(rec, [flt["song_mood"]], rng)


def plant(
    users: dict[int, list[dict]], questions: list[dict], places: PlaceDictionary, rng
) -> dict:
    planted: dict[str, list[int]] = {}
    protected = defaultdict(set)  # 사용자별로 다른 질문의 정답이라 건드리면 안 되는 기록
    for q in (
        questions
    ):  # 코멘트 의미 질문의 정답은 기록 속성과 무관하니 그대로 두되, 속성 변경은 피한다
        if "comment_gold" in q["filter"]:
            protected[q["user"]] |= comment_gold(q["filter"]["comment_gold"])
    for q in questions:
        if q.get("expect_none") or "comment_gold" in q["filter"]:
            continue
        records = users[q["user"]]
        gold = gold_of(q, records, places)
        need = 3 - len(gold)
        if need <= 0:
            continue
        window = DATE_WINDOW[q["user"]]
        others = [
            o
            for o in questions
            if o["user"] == q["user"]
            and o is not q
            and not o.get("expect_none")
            and "comment_gold" not in o["filter"]
        ]
        taken = set().union(*(gold_of(o, records, places) for o in others)) if others else set()
        pool = [r for r in records if r["record_id"] not in gold | taken | protected[q["user"]]]
        rng.shuffle(pool)
        chosen = pool[:need]
        for rec in chosen:
            _apply(rec, q["filter"], rng, places, window)
        planted[q["id"]] = [r["record_id"] for r in chosen]
    return planted


def validate(users, questions, places) -> list[str]:
    problems = []
    for q in questions:
        records = users[q["user"]]
        gold = gold_of(q, records, places)
        if q.get("expect_none"):
            if gold:
                problems.append(f"{q['id']}: 정답이 없어야 하는데 {len(gold)}건이 기준을 만족해요.")
        elif "comment_gold" in q["filter"]:
            if not gold:
                problems.append(f"{q['id']}: 코멘트 정답이 비어 있어요.")
        elif not 2 <= len(gold) <= 10:
            problems.append(f"{q['id']}: 정답 {len(gold)}건 (2~10건이어야 해요).")
        if "near" in q["filter"]:  # 반경 경계 근처 기록이 있으면 정답이 애매하다
            entry = next(e for e in places.entries if e.name == q["filter"]["near"])
            for r in records:
                d = haversine_km(r["lat"], r["lng"], entry.lat, entry.lng)
                if abs(d - entry.radius_km) < RADIUS_MARGIN_KM:
                    problems.append(
                        f"{q['id']}: 기록 {r['record_id']}가 반경 경계 근처예요({d:.2f}km)."
                    )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description="기억 검색 정확도 측정용 시험 데이터 생성")
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()

    questions = yaml.safe_load(QUESTIONS.read_text(encoding="utf-8"))
    places = PlaceDictionary.load()
    users = make_base(args.seed)
    planted = plant(users, questions, places, random.Random(args.seed + 1))
    problems = validate(users, questions, places)

    out_questions = []
    for q in questions:
        gold = sorted(gold_of(q, users[q["user"]], places))
        out_questions.append({**q, "gold": gold})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset.json").write_text(
        json.dumps(
            {
                "today": TODAY.isoformat(),
                "users": {str(u): recs for u, recs in users.items()},
                "questions": out_questions,
                "planted": planted,
            },
            ensure_ascii=False,
            indent=1,
            default=str,
        ),
        encoding="utf-8",
    )
    lines = [
        "# 기억 검색 측정 질문 초안",
        "",
        "| 번호 | 유형 | 사용자 | 질문 | 정답 기준 | 정답 건수 |",
        "|---|---|---|---|---|---|",
    ]
    for q in out_questions:
        flt = json.dumps(q["filter"], ensure_ascii=False, default=str)
        lines.append(
            f"| {q['id']} | {q['category']} | {q['user']} | {q['text']} | `{flt}` "
            f"| {len(q['gold'])} |"
        )
    lines += [
        "",
        "심은 기록(정답 건수를 맞추려고 속성을 바꾼 것): "
        + json.dumps(planted, ensure_ascii=False),
    ]
    (OUT_DIR / "questions_draft.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    sizes = {q["id"]: len(q["gold"]) for q in out_questions}
    print("질문별 정답 건수:", sizes)
    print("심은 기록:", {k: len(v) for k, v in planted.items()})
    print("문제:", problems or "없음")


if __name__ == "__main__":
    main()
