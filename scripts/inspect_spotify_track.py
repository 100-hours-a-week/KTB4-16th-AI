"""스포티파이에서 곡을 골랐을 때 실제로 어떤 정보가 오는지 확인한다.

사용법 (.env에 SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET 필요):
    python scripts/inspect_spotify_track.py "밤편지 아이유"      # 검색 → 1위 곡 상세
    python scripts/inspect_spotify_track.py --id 6rqhFgbbKwnb9MLmUQDhG6

출력:
    1) 검색 결과 목록 (백엔드 /api/music/search가 받는 것과 같은 원본)
    2) 곡 상세 원본 JSON  → fixtures/spotify_samples/track_<id>.json
    3) 아티스트 원본 JSON → fixtures/spotify_samples/artist_<id>.json (장르 확인용)
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.clients.spotify_client import SpotifyClient  # noqa: E402
from app.config import get_settings  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent.parent / "fixtures/spotify_samples"


def save(name: str, data: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", help="검색어 (예: '밤편지 아이유')")
    parser.add_argument("--id", help="스포티파이 곡 ID (검색 대신 바로 조회)")
    args = parser.parse_args()
    if not args.query and not args.id:
        parser.error("검색어 또는 --id 중 하나가 필요해요.")

    s = get_settings()
    if not s.spotify_client_id or not s.spotify_client_secret:
        sys.exit(".env에 SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET을 넣어 주세요.")
    client = SpotifyClient(s.spotify_client_id, s.spotify_client_secret, timeout=10)

    track_id = args.id
    if args.query:
        items = await client.search_tracks(args.query, limit=5)
        print(f"\n[검색 결과] '{args.query}' {len(items)}건")
        for i, t in enumerate(items, 1):
            artists = ", ".join(a["name"] for a in t["artists"])
            print(f"  {i}. {t['name']} - {artists}  (id={t['id']})")
        if not items:
            return
        track_id = items[0]["id"]

    track = await client.get_track(track_id)
    print(f"\n[곡 상세] 최상위 필드: {sorted(track.keys())}")
    print(f"  앨범 필드: {sorted(track['album'].keys())}")
    track_path = save(f"track_{track['id']}.json", track)
    print(f"  → {track_path.name}")

    artist = await client.get_artist(track["artists"][0]["id"])
    print(f"\n[아티스트] {artist['name']} 최상위 필드: {sorted(artist.keys())}")
    print(f"  장르: {artist.get('genres') or '(비어 있음)'}")
    artist_path = save(f"artist_{artist['id']}.json", artist)
    print(f"  → {artist_path.name}")


if __name__ == "__main__":
    asyncio.run(main())
