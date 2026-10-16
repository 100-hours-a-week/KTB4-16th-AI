"""곡 풀 CSV(개발/곡풀/04_TJ_곡풀.csv) → song_pool.json.

만든 파일은 공개 레포에 넣지 않고 비공개 HF 저장소(dionypark/mulo-song-pool)에 새 태그로 올린다.

설명·무드가 있는 곡만 넣는다. 노래방 순위·번호 같은 원본 차트 값은 넣지 않고, 차트에 오른
개월 수만 인기도(0~100)로 바꿔 넣는다.
실행: python scripts/build_song_pool.py <CSV 경로>  (현재 폴더에 song_pool.json 생성)
"""

import csv
import json
import sys
from pathlib import Path

OUT = Path("song_pool.json")
MAX_MONTHS = 12


def main(csv_path: str) -> None:
    songs = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            moods = [m.strip() for m in row["무드"].split(",") if m.strip()]
            if not row["설명"] or not moods:
                continue
            main_genre = row["주 장르"]
            others = [g.strip() for g in row["장르(전체)"].split(",") if g.strip()]
            genres = [main_genre] + [g for g in others if g != main_genre]
            months = int(row["TJ 차트 개월 수"] or 0)
            songs.append(
                {
                    "artist": row["가수"],
                    "title": row["곡"],
                    "genres": genres,
                    "moods": moods,
                    "description": row["설명"],
                    "popularity": round(min(months, MAX_MONTHS) / MAX_MONTHS * 100),
                }
            )
    OUT.write_text(
        json.dumps({"songs": songs}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    print(f"{OUT.name}: {len(songs)}곡")


if __name__ == "__main__":
    main(sys.argv[1])
