"""CLIP 벡터 비교 — Replicate와 RunPod CLIP 서버가 같은 사진에서 같은 벡터를 내는지 잰다.

폴더 안의 사진마다 두 곳에 같은 이미지(data URI)를 보내고 아래를 낸다.
결과는 CSV로 stdout에, 요약은 stderr에 나온다.

- 벡터 일치도: 두 벡터의 코사인 유사도. 1에 가까울수록 같은 벡터다. 서비스가 저장해 둔
  사진 벡터와 새 벡터를 섞어 쓰려면 이 값이 충분히 높아야 한다. 크기(norm)도 함께 낸다.
- 태그 일치: 같은 태그 분류 코드(ClipTagger)로 두 벡터를 분류해, 장소 1순위와
  다축 태그가 같은지 센다.
- 글 벡터 일치: app/tag_vectors.json에 저장된 Replicate 글 벡터(장소 태그)와 RunPod
  서버가 같은 문구에서 낸 벡터를 비교한다. 이 비교는 Replicate를 부르지 않는다.
- 속도: 호출 한 번의 시간. 네트워크와 대기가 섞여 있다. Replicate 쪽은 이미지
  인코딩만 잰 시간이다.

실행: PYTHONPATH=. .venv/bin/python scripts/photo_eval/clip_compare.py [사진 폴더] > 결과.csv
필요: .env의 REPLICATE_API_TOKEN, 환경변수 CLIP_API_KEY(RunPod CLIP 서버의 키),
      CLIP_SERVER_URL 환경변수 또는 --server-url(RunPod CLIP 서버 주소).
      --dry-run이면 Replicate 대신 가짜 CLIP을 쓴다(RunPod 서버는 --server-url 주소를 부른다).
Replicate를 사진 수만큼 부르므로 호출 비용이 든다.
공개 사진만 쓰고 개인 사진은 폴더에 넣지 않는다.
"""

import argparse
import asyncio
import csv
import math
import os
import statistics
import sys
import time
from pathlib import Path

import httpx

from app.components.tag_vector_file import read_vectors_file
from app.components.vector_math import cosine_similarity
from app.exceptions import ClipError
from scripts.photo_eval.clip_bench import (
    AXIS_COLUMNS,
    DEFAULT_FOLDER,
    IMAGE_SUFFIXES,
    build_client,
    build_tagger,
    data_uri,
)

# Pod 주소는 켤 때마다 바뀌고 공개 저장소에 둘 값이 아니라서 환경변수나 --server-url로 받는다
# (예: https://<PodID>-8001.proxy.runpod.net)
DEFAULT_SERVER_URL = os.environ.get("CLIP_SERVER_URL", "")
# 파이썬 기본 접속 표시는 RunPod 앞단에서 403(error code 1010)으로 막힌다
USER_AGENT = "muro-clip-compare/1.0"


def norm(vector: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vector))


def percentile_90(values: list[float]) -> float:
    return sorted(values)[int(0.9 * (len(values) - 1))]


def describe(values: list[float], digits: int = 4) -> str:
    return (
        f"최소 {min(values):.{digits}f} / 중앙값 {statistics.median(values):.{digits}f} / "
        f"평균 {statistics.mean(values):.{digits}f}"
    )


def axes_text(axes: dict[str, str]) -> str:
    return ";".join(f"{name}={axes[name]}" for name in AXIS_COLUMNS if name in axes)


async def text_parity(server: httpx.AsyncClient) -> list[float]:
    """저장된 Replicate 글 벡터(장소 태그)와 RunPod 서버가 같은 문구로 낸 벡터의 코사인 유사도."""
    stored = read_vectors_file()
    if stored is None:
        return []
    labels = list(stored.scene)
    res = await server.post("/encode_text", json={"texts": labels})
    res.raise_for_status()
    return [
        cosine_similarity(stored.scene[label], vector)
        for label, vector in zip(labels, res.json()["embeddings"], strict=True)
    ]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="RunPod CLIP 서버 주소")
    parser.add_argument("--dry-run", action="store_true", help="Replicate 대신 가짜 CLIP을 쓴다.")
    args = parser.parse_args()

    if not args.server_url:
        sys.exit("RunPod CLIP 서버 주소가 필요해요. --server-url이나 CLIP_SERVER_URL로 넣으세요.")
    key = os.environ.get("CLIP_API_KEY", "")
    if not key:
        sys.exit("CLIP_API_KEY 환경변수가 필요해요. RunPod CLIP 서버를 켤 때 쓴 키를 넣으세요.")
    replicate = build_client(args.dry_run)
    tagger = build_tagger(replicate)
    photos = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        sys.exit(f"{args.folder}에 사진(jpg·png)이 없어요.")

    out = csv.writer(sys.stdout)
    out.writerow(
        [
            "photo",
            "replicate_seconds",
            "runpod_seconds",
            "cosine",
            "norm_replicate",
            "norm_runpod",
            "scene_replicate",
            "scene_runpod",
            "scene_top1_same",
            "axes_replicate",
            "axes_runpod",
            "axes_same",
        ]
    )
    cosines: list[float] = []
    replicate_seconds: list[float] = []
    runpod_seconds: list[float] = []
    scene_same = axes_same = 0

    headers = {"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT}
    async with httpx.AsyncClient(base_url=args.server_url, headers=headers, timeout=60) as server:
        try:
            parity = await text_parity(server)
        except httpx.HTTPError as e:
            sys.exit(f"RunPod CLIP 서버에 연결하지 못했어요: {type(e).__name__} {e}")
        if parity:
            print(
                f"# 글 벡터 코사인(장소 태그 {len(parity)}개): {describe(parity)}", file=sys.stderr
            )

        for path in photos:
            uri = data_uri(path)
            started = time.perf_counter()
            try:
                vector_replicate = await replicate.encode_image(uri)
            except ClipError as e:
                out.writerow([path.name, f"ERROR Replicate: {e}"])
                continue
            elapsed_replicate = time.perf_counter() - started

            started = time.perf_counter()
            try:
                res = await server.post("/encode_image", json={"image": uri})
                res.raise_for_status()
                vector_runpod = res.json()["embedding"]
            except (httpx.HTTPError, KeyError) as e:
                out.writerow([path.name, f"ERROR RunPod: {type(e).__name__} {e}"])
                continue
            elapsed_runpod = time.perf_counter() - started

            cosine = cosine_similarity(vector_replicate, vector_runpod)
            scene_replicate = await tagger.tag_from_vector(vector_replicate, at_least_one=True)
            scene_runpod = await tagger.tag_from_vector(vector_runpod, at_least_one=True)
            axes_replicate = await tagger.classify_axes(vector_replicate)
            axes_runpod = await tagger.classify_axes(vector_runpod)

            top1_same = scene_replicate[:1] == scene_runpod[:1]
            same_axes = axes_replicate == axes_runpod
            scene_same += top1_same
            axes_same += same_axes
            cosines.append(cosine)
            replicate_seconds.append(elapsed_replicate)
            runpod_seconds.append(elapsed_runpod)
            out.writerow(
                [
                    path.name,
                    f"{elapsed_replicate:.2f}",
                    f"{elapsed_runpod:.2f}",
                    f"{cosine:.4f}",
                    f"{norm(vector_replicate):.2f}",
                    f"{norm(vector_runpod):.2f}",
                    "|".join(scene_replicate),
                    "|".join(scene_runpod),
                    top1_same,
                    axes_text(axes_replicate),
                    axes_text(axes_runpod),
                    same_axes,
                ]
            )

    if cosines:
        n = len(cosines)
        print(f"# 사진 {n}장 이미지 벡터 코사인: {describe(cosines)}", file=sys.stderr)
        print(
            f"# 장소 1순위 같음 {scene_same}/{n}, 다축 태그 전체 같음 {axes_same}/{n}",
            file=sys.stderr,
        )
        for name, values in (("Replicate", replicate_seconds), ("RunPod", runpod_seconds)):
            print(
                f"# {name} 시간: 중앙값 {statistics.median(values):.2f}s "
                f"p90 {percentile_90(values):.2f}s",
                file=sys.stderr,
            )


if __name__ == "__main__":
    asyncio.run(main())
