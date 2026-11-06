"""상황 또는 사진 → 장면 3개와 장면마다 문장 2종(가사 문체·설명 문체) — 기능1·3 공통.

곡을 고르기 전에 "이 상황에서 사람들이 있을 법한 장면"을 로컬 LLM(RunPod Gemma)이 서로 다르게
3개 상상하고, 장면마다 두 문장을 쓴다.

  lyric       — 그 장면의 사람이 속으로 하는 말을 노래 가사처럼 4줄 (가사 벡터와 비교)
  description — 곡 소개 문체 3문장, 첫 문장은 "곡이다."로 끝 (곡 설명 벡터와 비교)

곡 쪽 벡터가 가사(lyric_embedding)와 곡 설명(description_embedding) 두 가지라서, 질문 문장도 각
벡터와 같은 문체로 쓰면 더 잘 맞는다 (100개 상황 실측: 같은 문체로 맞춘 방식 3.82점,
설명 문장 하나를 가사 벡터와 비교한 방식 3.55점 — 곡풀/README 참고).

장르·무드 태그와 곡 이름은 받지 않는다. 곡은 곡풀에서 벡터로만 고르고, 취향 장르는 별도로
거른다(taste_genres.py).
"""

import json
import logging
import re
from dataclasses import dataclass

from app.clients.llm_client import LLMClient
from app.exceptions import LLMError

logger = logging.getLogger("muro.scene_curator")

PROMPT_VERSION = "scene-v1"
SCENE_COUNT = 3
MAX_TOKENS = 1200
# JSON이 깨져 오는 경우가 가끔 있어서 한 번 더 요청한다
ATTEMPTS = 2

_FORMAT = """아래 JSON 하나만 출력해.
{"scenes": [{"scene": "...", "description": "...", "lyric": "..."}, ...]}
- scene: 누가 어디서 무엇을 하는지 한 문장
- description: 정확히 3문장. 첫 문장은 그 장면의 주제와 감정을 쓰고 "곡이다."로 끝내.
  두 번째와 세 번째 문장은 그 장면의 사람이 속으로 하는 이야기와 마음을 노래 가사 내용처럼 써.
- lyric: 그 장면의 사람이 속으로 하는 말을 노래 가사처럼 한국어로 4줄 써. 줄은 " / "로 이어.
  실제 노래 가사를 베끼지 말고 새로 써.
- 세 장면은 감정과 내용이 서로 달라야 해."""

CONTEXT_SYSTEM = (
    "아래는 지금 상황이야. 이 계절·날씨·시간에 사람들이 있을 법한 장면을 서로 다르게 3개 상상해. "
    + _FORMAT
    + "\n- 세 장면 모두 상황의 계절·날씨·시간대·평일/주말과 맞아야 해."
)

PHOTO_SYSTEM = (
    "이 사진을 보고, 사진에 담긴 순간에 사람이 있을 법한 장면을 서로 다르게 3개 상상해. "
    + _FORMAT
    + "\n- 세 장면 모두 사진에 실제로 보이는 것(장소·날씨·분위기·사람)과 맞아야 해."
    + "\n- 사진에 보이지 않는 것을 지어내지 마."
)
PHOTO_PROMPT = "위 사진을 보고 장면을 만들어 줘."


@dataclass(frozen=True)
class Scene:
    scene: str
    description: str
    lyric: str


class SceneCurator:
    def __init__(self, llm: LLMClient):
        self._llm = llm

    @property
    def model(self) -> str:
        return self._llm.model

    async def for_context(self, situation: str) -> list[Scene]:
        """기능1 — 계절·요일·시간대·날씨·기온이 담긴 상황 문장으로."""
        if not situation.strip():
            raise LLMError("상황 설명이 없으면 장면을 만들 수 없어요.")
        return await self._scenes(CONTEXT_SYSTEM, f"상황: {situation}")

    async def for_photo(self, image_url: str) -> list[Scene]:
        """기능3 — 사진을 모델이 직접 보고 (CLIP 태그 없이). image_url은 https 주소나 data: URL."""
        if not image_url:
            raise LLMError("사진이 없으면 장면을 만들 수 없어요.")
        return await self._scenes(PHOTO_SYSTEM, PHOTO_PROMPT, image_url)

    async def _scenes(self, system: str, prompt: str, image_url: str | None = None) -> list[Scene]:
        if not prompt.strip():
            raise LLMError("상황 설명이 없으면 장면을 만들 수 없어요.")
        last = "형식 불일치"
        for _ in range(ATTEMPTS):
            text = await self._llm.complete(
                system=system,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                json_mode=True,
                image_url=image_url,
            )
            scenes = parse_scenes(text)
            if scenes:
                return scenes
            logger.warning("장면 응답을 해석하지 못함, 다시 요청: %s", text[:120])
        raise LLMError(f"장면을 만들지 못했어요 ({last})")


def parse_scenes(text: str) -> list[Scene]:
    """JSON에서 문장 둘이 다 있는 장면만 남긴다. 하나도 없으면 빈 목록."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        raw = json.loads(match.group(0)).get("scenes", [])
    except (ValueError, AttributeError):
        return []
    scenes: list[Scene] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        scene = str(item.get("scene") or "").strip()
        description = str(item.get("description") or "").strip()
        lyric = str(item.get("lyric") or "").strip()
        if scene and description and lyric:
            scenes.append(Scene(scene, description, lyric))
    return scenes[:SCENE_COUNT]
