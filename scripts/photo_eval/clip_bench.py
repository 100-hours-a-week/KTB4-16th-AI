"""사진 평가 세트의 CLIP 태깅 속도·결과 측정 — 서비스가 쓰는 CLIP 코드(Replicate)를 그대로 돌린다.

폴더 안의 사진마다 CLIP 이미지 인코딩(Replicate 호출) 시간을 재고, 서비스와 같은 방식으로
장소 태그와 다축 태그(사진 종류·대상·분위기·표정·감정)를 낸다. 결과는 CSV로 stdout에 나온다.

- seconds는 Replicate 호출(이미지 인코딩)만 잰 시간이다. 대기·시작·1초 간격 조회가 섞이고 집에서
  Replicate까지의 네트워크도 들어가므로, VLM 서버 시간과 같은 조건이 아니다.
- 태그는 미리 계산한 app/tag_vectors.json을 쓴다. 이 파일이 없거나 CLIP_MODEL과 버전이 다르면
  태그 벡터를 CLIP으로 새로 계산해야 해서(몇 분, 약 52번 호출) 측정하지 않고 멈춘다.
- 사진은 data URI로 Replicate에 보낸다. 공개 사진만 쓰고, 개인 사진은 폴더에 넣지 않는다.

실행: PYTHONPATH=. .venv/bin/python scripts/photo_eval/clip_bench.py [사진 폴더] > 결과.csv
      --dry-run이면 Replicate 대신 가짜 CLIP으로 돌려 흐름만 확인한다(외부 호출·비용 없음).
필요: .env의 REPLICATE_API_TOKEN (--dry-run은 불필요)
"""

import argparse
import asyncio
import base64
import csv
import statistics
import sys
import time
from pathlib import Path

from app.clients.clip_client import ClipClient, FakeClipClient, ReplicateClipClient
from app.components.axis_vectors import AxisVectors
from app.components.category_vectors import CategoryVectors
from app.components.clip_tagger import ClipTagger
from app.components.tag_vector_file import read_vectors_file
from app.components.tag_version import compute_tag_version
from app.config import get_settings
from app.exceptions import ClipError

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FOLDER = ROOT / "data/photo_eval/images"
AXIS_COLUMNS = ["사진 종류", "대상", "분위기", "표정·감정"]  # app/axis_tags.yaml의 축 이름
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def data_uri(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def build_client(dry_run: bool) -> ClipClient:
    s = get_settings()
    if dry_run:
        return FakeClipClient(dim=s.clip_embedding_dim, model=s.clip_model)
    if not s.replicate_api_token:
        sys.exit("REPLICATE_API_TOKEN이 비어 있어요. .env를 확인하세요.")
    return ReplicateClipClient(
        api_token=s.replicate_api_token,
        model=s.clip_model,
        dim=s.clip_embedding_dim,
        timeout=s.external_timeout_seconds,
        max_retries=s.external_max_retries,
    )


def build_tagger(client: ClipClient) -> ClipTagger:
    vectors = read_vectors_file()
    if vectors is None or vectors.version != compute_tag_version(client.model):
        sys.exit(
            "app/tag_vectors.json이 없거나 지금 태그 정의·CLIP_MODEL과 버전이 달라요. "
            "scripts/build_tag_vectors.py로 다시 만든 뒤 실행하세요."
        )
    return ClipTagger(
        client,
        CategoryVectors(client, precomputed=vectors),
        AxisVectors(client, precomputed=vectors),
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("--dry-run", action="store_true", help="가짜 CLIP으로 흐름만 확인한다.")
    args = parser.parse_args()

    tagger = build_tagger(build_client(args.dry_run))
    photos = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        sys.exit(f"{args.folder}에 사진(jpg·png)이 없어요.")

    out = csv.writer(sys.stdout)
    out.writerow(["photo", "seconds", "scene", *AXIS_COLUMNS])
    seconds: list[float] = []
    for path in photos:
        started = time.perf_counter()
        try:
            vector = await tagger.encode_image(data_uri(path))
        except ClipError as e:
            out.writerow([path.name, f"ERROR {e}"])
            continue
        elapsed = time.perf_counter() - started
        scene = await tagger.tag_from_vector(vector, at_least_one=True)
        axes = await tagger.classify_axes(vector)
        seconds.append(elapsed)
        out.writerow(
            [path.name, f"{elapsed:.2f}", "|".join(scene), *(axes.get(c, "") for c in AXIS_COLUMNS)]
        )

    if seconds:
        ordered = sorted(seconds)
        p90 = ordered[int(0.9 * (len(ordered) - 1))]
        print(
            f"# n={len(ordered)} median={statistics.median(ordered):.2f}s p90={p90:.2f}s",
            file=sys.stderr,
        )


if __name__ == "__main__":
    asyncio.run(main())
