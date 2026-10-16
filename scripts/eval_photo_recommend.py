"""기능3 사진 추천 평가 — 같은 사진으로 "LLM 5곡"과 "곡 풀 3 + LLM 2"를 비교한다.

사진: 2026-10-04 CLIP 다축 실험 사진 32장 (위키미디어 공개 라이선스, photos.json + img/)
공정하게 비교하려고 사진 분석(CLIP)은 사진당 한 번만 하고 두 방식에 똑같이 넣는다.
Spotify 확인 결과는 실행 동안 메모리에 캐시해서 같은 곡을 두 번 찾지 않는다.

자동으로 재는 것 (방식별)
  - 응답 곡 수, 곡 풀 곡 비율
  - 실존율: Spotify에서 찾아본 곡 중 실제로 있던 비율 (LLM이 지어낸 곡은 여기서 걸린다)
  - 다양성: 전체 응답 곡 중 서로 다른 곡 비율 (같은 곡이 여러 사진에 나오면 낮아짐)
  - Spotify 호출 수, LLM 호출 수, 사진당 처리 시간
사람이 판정할 것: 결과 CSV의 fits (O/X) — 방식 칸을 가리고 보면 편향이 줄어든다.

Spotify 하루 한도 사고(10/02)가 있어서 429가 나오면 즉시 멈추고, 곡 확인 사이에 쉰다.
실행: python scripts/eval_photo_recommend.py <사진 폴더> [결과 CSV] [--limit N]
"""

import argparse
import asyncio
import base64
import csv
import json
import time
from pathlib import Path

from app.components.clip_tagger import PhotoAnalysis
from app.components.reranker import Reranker
from app.components.song_curator import CANDIDATE_COUNT
from app.components.song_pool import load_song_pool
from app.config import get_settings
from app.dependencies import get_clip_tagger, get_song_curator, get_spotify_search_client
from app.schemas.photo_recommend import PhotoRecommendRequest
from app.services.photo_recommend_service import PhotoRecommendService

MODES = {
    "llm5": {"pool_count": 0, "llm_count": CANDIDATE_COUNT},
    "pool3_llm2": {"pool_count": 3, "llm_count": 4},
}
SPOTIFY_PAUSE_SECONDS = 0.5


class StopForQuota(Exception):
    pass


class CountingSpotify:
    """호출 수를 세고, 429면 평가를 멈춘다."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    async def search_tracks(self, query: str, limit: int = 10):
        self.calls += 1
        await asyncio.sleep(SPOTIFY_PAUSE_SECONDS)
        try:
            return await self._inner.search_tracks(query, limit)
        except Exception as e:
            if "429" in str(e):
                raise StopForQuota(str(e)) from e
            raise

    async def get_track(self, track_id: str):
        return await self._inner.get_track(track_id)

    async def get_artist(self, artist_id: str):
        return await self._inner.get_artist(artist_id)


class MemoryLookups:
    def __init__(self):
        self.rows = {}

    async def get_many(self, keys):
        return {k: self.rows[k] for k in keys if k in self.rows}

    async def save_many(self, entries):
        self.rows |= {e.lookup_key: e for e in entries}


class FixedClip:
    def __init__(self, analysis: PhotoAnalysis):
        self._analysis = analysis

    async def analyze(self, image_url: str, top_k: int = 3) -> PhotoAnalysis:
        return self._analysis


class CountingCurator:
    def __init__(self, inner):
        self._inner = inner
        self.calls = 0
        self.last = None

    async def curate_from_tags(self, tags, exclude=None, axes=None, count=CANDIDATE_COUNT):
        self.calls += 1
        result = await self._inner.curate_from_tags(tags, exclude=exclude, axes=axes, count=count)
        self.last = result
        return result


def data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode()


async def run(photo_dir: Path, out_path: Path, limit: int | None) -> None:
    photos = json.loads((photo_dir / "photos.json").read_text(encoding="utf-8"))[:limit]
    clip = get_clip_tagger()
    curator = CountingCurator(get_song_curator())
    spotify = CountingSpotify(get_spotify_search_client())
    lookups = MemoryLookups()
    pool_path = Path(get_settings().song_pool_path)
    pool = load_song_pool(pool_path)
    reranker = Reranker()
    pool_keys = (
        {
            (s["artist"].lower(), s["title"].lower())
            for s in json.loads(pool_path.read_text(encoding="utf-8"))["songs"]
        }
        if pool_path.exists()
        else set()
    )

    rows = []
    stats = {
        m: {"tracks": 0, "pool": 0, "ids": [], "seconds": [], "spotify": 0, "llm": 0} for m in MODES
    }
    try:
        for photo in photos:
            img = next((photo_dir / "img").glob(f"{photo['id']}.*"), None)
            if img is None:
                continue
            analysis = await clip.analyze(data_uri(img))
            for mode, cfg in MODES.items():
                service = PhotoRecommendService(
                    FixedClip(analysis),
                    curator,
                    reranker,
                    spotify,
                    lookups,
                    song_pool=pool,
                    **cfg,
                )
                s = stats[mode]
                before_spotify, before_llm = spotify.calls, curator.calls
                started = time.perf_counter()
                req = PhotoRecommendRequest(imageUrl="https://eval.local/x.jpg")
                res = await service.recommend(req)
                s["seconds"].append(time.perf_counter() - started)
                s["spotify"] += spotify.calls - before_spotify
                s["llm"] += curator.calls - before_llm
                for t in res.tracks:
                    from_pool = (t.artist_name.lower(), t.title.lower()) in pool_keys
                    s["tracks"] += 1
                    s["pool"] += from_pool
                    s["ids"].append(t.external_track_id)
                    rows.append(
                        {
                            "photo": photo["id"],
                            "intended": photo.get("intended", ""),
                            "scene_tags": "|".join(res.mood_tags),
                            "mode": mode,
                            "artist": t.artist_name,
                            "title": t.title,
                            "from_pool": int(from_pool),
                            "fits (O/X)": "",
                        }
                    )
                print(f"{photo['id']} {mode}: {[t.title for t in res.tracks]}")
    except StopForQuota as e:
        print(f"\nSpotify 429 — 평가를 멈춤: {e}")

    if rows:
        with out_path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    checked = list(lookups.rows.values())
    exists = sum(e.track is not None for e in checked)
    print(f"\nLLM {get_settings().llm_model}, 사진 {len({r['photo'] for r in rows})}장")
    print(f"실존율(두 방식 합산) {exists}/{len(checked)}")
    for mode, s in stats.items():
        n = s["tracks"] or 1
        photos_done = len(s["seconds"]) or 1
        print(
            f"[{mode}] 곡 {s['tracks']}, 곡 풀 비율 {s['pool'] / n:.0%}, "
            f"다양성 {len(set(s['ids'])) / n:.0%}, "
            f"Spotify 호출 {s['spotify']}, LLM 호출 {s['llm']}, "
            f"사진당 {sum(s['seconds']) / photos_done:.1f}s"
        )
    print(f"사람 판정: {out_path}의 fits 칸 (mode 칸은 가리고 보기)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("photo_dir", type=Path)
    parser.add_argument("out", nargs="?", default="eval_photo_recommend_result.csv")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    asyncio.run(run(args.photo_dir, Path(args.out), args.limit))
