"""태그 벡터를 미리 계산해 app/tag_vectors.json에 저장한다.

장소 태그(app/mood_tags.yaml)와 다축 태그(app/axis_tags.yaml)의 문장을 CLIP으로 벡터로 바꾼다.
Replicate를 문장마다 한 번씩 순서대로 부르므로 몇 분(약 52번 호출)이 걸린다. 서버는 이 파일을
읽기만 해서 켜진 뒤 첫 요청이 느려지지 않는다.

태그 정의(yaml)를 바꾸거나 CLIP 모델을 바꿀 때마다 다시 실행해서 파일을 새로 만들고 커밋한다.
서버에 쓰이는 CLIP_MODEL 값과 같은 값으로 실행해야 한다(다르면 서버가 파일을 쓰지 않는다).

실행: PYTHONPATH=. .venv/bin/python scripts/build_tag_vectors.py
필요: .env의 AI_CLIENT_MODE=real, REPLICATE_API_TOKEN
"""

import asyncio
import sys
import time

from app.clients.clip_client import ReplicateClipClient
from app.components.axis_vectors import AxisVectors, load_axis_specs
from app.components.category_vectors import CategoryVectors, load_tag_labels
from app.components.tag_vector_file import DEFAULT_VECTORS_PATH, from_loaded, write_vectors_file
from app.components.tag_version import compute_tag_version
from app.config import get_settings


class ProgressClip:
    """문장 하나씩 인코딩하면서 진행 상황을 출력한다. 결과는 한 번에 부른 것과 같다."""

    def __init__(self, inner: ReplicateClipClient, total: int):
        self._inner = inner
        self._total = total
        self._done = 0
        self.model = inner.model
        self.dim = inner.dim

    async def encode_text(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vectors.extend(await self._inner.encode_text([text]))
            self._done += 1
            print(f"[{self._done}/{self._total}] {text[:50]}", flush=True)
        return vectors


async def main() -> None:
    settings = get_settings()
    if settings.ai_client_mode == "fake":
        sys.exit("AI_CLIENT_MODE가 fake라서 가짜 벡터가 만들어져요. real로 바꿔서 실행하세요.")
    if not settings.replicate_api_token:
        sys.exit("REPLICATE_API_TOKEN이 없어요.")

    real = ReplicateClipClient(
        api_token=settings.replicate_api_token,
        model=settings.clip_model,
        dim=settings.clip_embedding_dim,
        timeout=settings.external_timeout_seconds,
        max_retries=settings.external_max_retries,
    )
    total = len(load_tag_labels(CategoryVectors(real)._path)) + sum(
        len(tag.sentences) for spec in load_axis_specs(AxisVectors(real)._path) for tag in spec.tags
    )
    clip = ProgressClip(real, total)
    print(f"CLIP 모델: {real.model} (차원 {real.dim}), 인코딩할 문장 {total}개", flush=True)

    started = time.monotonic()
    categories, axes = CategoryVectors(clip), AxisVectors(clip)  # 미리 계산한 파일은 넣지 않는다
    await categories.load()
    await axes.load()

    version = compute_tag_version(real.model)
    write_vectors_file(DEFAULT_VECTORS_PATH, from_loaded(version, real.model, categories, axes))
    print(f"저장 완료: {DEFAULT_VECTORS_PATH} (버전 {version}, {time.monotonic() - started:.0f}초)")


if __name__ == "__main__":
    asyncio.run(main())
