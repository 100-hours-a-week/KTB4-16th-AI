"""곡 풀 곡들의 Spotify ID를 하루 치씩 찾아 둔다 (곡 풀을 track_moods와 합칠 때 쓸 재료).

Spotify 하루 한도를 정확히 몰라서 한 번에 --limit곡(기본 100)만 하고,
에러가 한 번이라도 나면 그날은 바로 멈춘다. 결과는 --state 파일에 쌓이고,
이미 처리한 곡은 건너뛰어서 다시 돌리면 이어서 한다.
  - 찾음: {"id", "artist", "title"}  — Spotify가 돌려준 가수·제목(검수용)
  - 못 찾음: {"id": null}           — 가수 이름이 맞는 결과가 없을 때 (나중에 사람이 본다)

실행 (앱 설정의 SPOTIFY_CLIENT_ID/SECRET을 쓴다):
    python scripts/resolve_pool_spotify_ids.py <song_pool.json> <state.json> [--limit 100]
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

from app.clients.spotify_client import SpotifyClient, SpotifyError
from app.config import get_settings

DELAY_SECONDS = 2.0


def key_of(song: dict) -> str:
    return f"{song['artist']} || {song['title']}"


def norm(text: str) -> str:
    return re.sub(r"[\W_]+", "", text).lower()


def pick(items: list[dict], artist: str) -> dict | None:
    want = norm(artist)
    for item in items:
        names = [norm(a["name"]) for a in item.get("artists", [])]
        if any(want and (want in n or n in want) for n in names):
            return item
    return None


async def main(pool_path: str, state_path: str, limit: int) -> None:
    s = get_settings()
    spotify = SpotifyClient(s.spotify_client_id, s.spotify_client_secret, timeout=15)
    songs = json.loads(Path(pool_path).read_text(encoding="utf-8"))["songs"]
    state_file = Path(state_path)
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.exists() else {}

    todo = [song for song in songs if key_of(song) not in state][:limit]
    print(f"처리 {len(state)}/{len(songs)}곡 — 오늘 {len(todo)}곡")
    done = 0
    for song in todo:
        title = song["title"].replace('"', "")
        artist = song["artist"].replace('"', "")
        try:
            items = await spotify.search_tracks(f'track:"{title}" artist:"{artist}"', limit=5)
        except SpotifyError as e:
            print(f"중단: {e}")
            break
        hit = pick(items, song["artist"])
        state[key_of(song)] = (
            {"id": hit["id"], "artist": hit["artists"][0]["name"], "title": hit["name"]}
            if hit
            else {"id": None}
        )
        done += 1
        state_file.write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        await asyncio.sleep(DELAY_SECONDS)

    found = sum(1 for v in state.values() if v["id"])
    print(f"오늘 {done}곡 처리, 누적 {len(state)}/{len(songs)}곡 (ID 찾음 {found})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pool")
    ap.add_argument("state")
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()
    asyncio.run(main(a.pool, a.state, a.limit))
