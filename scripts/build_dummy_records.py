"""백엔드 자물쇠 테스트 케이스 → /api/embeddings/generate 요청 형식으로 변환.

입력: fixtures/raw/backend_record_cases.json (백엔드 POST /api/records 테스트 케이스)
출력: fixtures/dummy_records.json

- case == "valid" 인 항목의 seed(DB에 저장되는 자물쇠)만 사용
- 사진은 필수이므로 photoUrl이 없는 seed는 제외

사용법: python scripts/build_dummy_records.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "fixtures/raw/backend_record_cases.json"
DST = ROOT / "fixtures/dummy_records.json"


def to_request(seed: dict) -> dict:
    track = seed["track"]
    return {
        "recordId": seed["recordId"],
        "userId": seed["userId"],
        "photoUrl": seed["photoUrl"],
        "track": {
            "externalTrackId": track.get("externalTrackId"),
            "title": track["title"],
            "artistName": track.get("artistName") or track["artist"],
            "genre": track.get("genre"),
        },
        "comment": seed.get("comment"),
        "createdAt": seed["createdAt"],
    }


def main() -> None:
    cases = json.loads(SRC.read_text(encoding="utf-8"))
    seeds = [c["seed"] for c in cases if c["case"] == "valid" and c.get("seed")]
    records = [to_request(s) for s in seeds if s.get("photoUrl")]
    DST.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(seeds)}건 중 사진 있는 {len(records)}건 → {DST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
