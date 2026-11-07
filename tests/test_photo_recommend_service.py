import random

import pytest

from app.clients.image_fetcher import ImageFetchError
from app.clients.spotify_client import SpotifyError
from app.exceptions import ExternalApiFailedError, LLMError
from app.schemas.photo_recommend import PhotoRecommendRequest
from app.services.photo_recommend_service import PhotoRecommendService
from tests.scene_fakes import (
    SCENES,
    StubEmbedder,
    StubImages,
    StubSceneCurator,
    StubSpotify,
    pool_of,
    song,
)

REQ = PhotoRecommendRequest(imageUrl="https://cdn.muro.app/uploads/img_1.jpg")


def _pool(count=8):
    return pool_of([song(f"가수{i}", f"곡{i}", vector=(1.0, i * 0.05, 0.0)) for i in range(count)])


def _service(*, pool, images=None, curator=None, spotify=None, embedder=None, **extra):
    songs = pool._songs if pool is not None else []
    return PhotoRecommendService(
        scene_curator=curator or StubSceneCurator(),
        spotify_search=spotify or StubSpotify(songs),
        embedder=embedder or StubEmbedder(),
        image_fetcher=images or StubImages(),
        song_pool=pool,
        rng=random.Random(0),
        **extra,
    )


async def test_recommends_five_pool_songs_with_a_scene_reason():
    res = await _service(pool=_pool()).recommend(REQ)

    assert res.degraded is False
    assert res.mood_tags == []  # CLIP 태그를 뺐다
    assert len(res.tracks) == 5
    assert len({t.external_track_id for t in res.tracks}) == 5
    assert res.recommend_reason == "노을 지는 바닷가 사진이라 발라드 위주로 골라봤어요."


async def test_reason_falls_back_when_the_photo_summary_is_missing():
    res = await _service(pool=_pool(), curator=StubSceneCurator(summary="")).recommend(REQ)

    assert res.recommend_reason == "사진 분위기에 맞춰 발라드 위주로 골라봤어요."


async def test_the_downloaded_photo_goes_to_the_scene_curator():
    images, curator = StubImages(), StubSceneCurator()

    await _service(pool=_pool(), images=images, curator=curator).recommend(REQ)

    assert images.urls == [REQ.image_url]
    assert curator.situations == ["data:image/jpeg;base64,AAAA"]


async def test_each_scene_sentence_is_embedded_once_in_one_batch():
    embedder = StubEmbedder()

    await _service(pool=_pool(), embedder=embedder).recommend(REQ)

    assert embedder.batches == [[s.lyric for s in SCENES] + [s.description for s in SCENES]]


async def test_photo_download_failure_falls_back():
    res = await _service(pool=_pool(), images=StubImages(error=ImageFetchError("x"))).recommend(REQ)

    assert res.degraded is True
    assert res.tracks == []


async def test_scene_failure_falls_back():
    curator = StubSceneCurator(error=LLMError("down"))

    res = await _service(pool=_pool(), curator=curator).recommend(REQ)

    assert res.degraded is True
    assert res.recommend_reason is None


async def test_embedding_failure_falls_back():
    res = await _service(pool=_pool(), embedder=StubEmbedder(error=RuntimeError("down"))).recommend(
        REQ
    )

    assert res.degraded is True


async def test_missing_pool_falls_back_without_downloading_the_photo():
    images = StubImages()

    res = await _service(pool=None, images=images).recommend(REQ)

    assert res.degraded is True
    assert images.urls == []


async def test_spotify_down_raises_external_api_failed():
    pool = _pool()
    spotify = StubSpotify(pool._songs, error=SpotifyError("Spotify API 오류 (status 429)"))

    with pytest.raises(ExternalApiFailedError):
        await _service(pool=pool, spotify=spotify).recommend(REQ)


async def test_all_songs_missing_on_spotify_is_an_empty_result_not_an_error():
    res = await _service(pool=_pool(), spotify=StubSpotify([])).recommend(REQ)

    assert res.degraded is False
    assert res.tracks == []
    assert res.recommend_reason is None


async def test_fewer_pool_songs_than_five_returns_what_exists():
    res = await _service(pool=_pool(count=3)).recommend(REQ)

    assert len(res.tracks) == 3


async def test_same_artist_appears_once():
    pool = pool_of([song("아이유", f"곡{i}", vector=(1.0, i * 0.01, 0.0)) for i in range(6)])

    res = await _service(pool=pool).recommend(REQ)

    assert len(res.tracks) == 1


async def test_hub_weight_keeps_songs_that_score_high_everywhere_out_of_the_candidates():
    """허브 값이 큰 곡(어디서나 높은 곡)은 점수가 깎여 상위 후보에서 빠진다."""
    pool = pool_of(
        [
            song("허브가수", "허브곡", vector=(1.0, 0.0, 0.0), hub=10.0),
            *[song(f"가수{i}", f"곡{i}", vector=(0.9, 0.1 + i * 0.01, 0.0)) for i in range(30)],
        ]
    )

    corrected = await _service(pool=pool, hub_weight=1.0).recommend(REQ)

    assert "허브곡" not in [t.title for t in corrected.tracks]


async def test_five_tracks_are_still_returned_when_spotify_misses_some_picks():
    pool = _pool(count=30)
    ranked = pool.rank_scenes([[1, 0, 0]], [[1, 0, 0]])
    missing = {r.song.title for r in ranked[:10:2]}  # 상위 후보의 일부는 Spotify에 없다
    found = [s for s in pool._songs if s.title not in missing]

    res = await _service(pool=pool, spotify=StubSpotify(found)).recommend(REQ)

    assert len(res.tracks) == 5
    assert not missing & {t.title for t in res.tracks}
