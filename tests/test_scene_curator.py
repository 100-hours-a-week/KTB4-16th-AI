import json

import pytest

from app.components.scene_curator import (
    CONTEXT_SYSTEM,
    PHOTO_SYSTEM,
    SceneCurator,
    parse_scenes,
)
from app.exceptions import LLMError


def _scene(n: int, **override) -> dict:
    scene = {
        "scene": f"장면{n}",
        "description": f"설명{n} 곡이다. 가. 나.",
        "lyric": f"가사{n} / 둘 / 셋 / 넷",
    }
    scene.update(override)
    return scene


GOOD = json.dumps({"scenes": [_scene(1), _scene(2), _scene(3)]}, ensure_ascii=False)


class ScriptedLLM:
    model = "scripted"

    def __init__(self, *responses: str):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def complete(self, *, system, prompt, max_tokens=1024, json_mode=False, image_url=None):
        self.calls.append(
            {"system": system, "prompt": prompt, "json_mode": json_mode, "image_url": image_url}
        )
        return self._responses.pop(0)


def test_parse_scenes_reads_three_complete_scenes():
    scenes = parse_scenes(GOOD)

    assert [s.scene for s in scenes] == ["장면1", "장면2", "장면3"]
    assert scenes[0].lyric.startswith("가사1")


def test_parse_scenes_drops_scenes_missing_a_sentence_and_keeps_at_most_three():
    raw = json.dumps(
        {"scenes": [_scene(1), _scene(2, lyric=""), _scene(3), _scene(4), _scene(5), "x"]},
        ensure_ascii=False,
    )

    assert [s.scene for s in parse_scenes(raw)] == ["장면1", "장면3", "장면4"]


@pytest.mark.parametrize("raw", ["", "not json", '{"scenes": "x"}', '{"scenes": []}', "[1, 2]"])
def test_parse_scenes_returns_empty_for_broken_output(raw):
    assert parse_scenes(raw) == []


def test_parse_scenes_finds_json_inside_extra_text():
    assert len(parse_scenes(f"여기 있어요\n{GOOD}\n끝")) == 3


async def test_for_context_asks_json_with_the_situation():
    llm = ScriptedLLM(GOOD)

    scenes = await SceneCurator(llm).for_context("여름(8월), 평일 저녁, 비 오는 날 (기온 18도)")

    assert len(scenes) == 3
    assert llm.calls[0]["system"] == CONTEXT_SYSTEM
    assert llm.calls[0]["prompt"] == "상황: 여름(8월), 평일 저녁, 비 오는 날 (기온 18도)"
    assert llm.calls[0]["json_mode"] is True


async def test_for_photo_sends_the_photo_to_the_model():
    llm = ScriptedLLM(GOOD)

    scenes = await SceneCurator(llm).for_photo("data:image/jpeg;base64,AAAA")

    assert len(scenes) == 3
    assert llm.calls[0]["system"] == PHOTO_SYSTEM
    assert llm.calls[0]["image_url"] == "data:image/jpeg;base64,AAAA"
    assert llm.calls[0]["json_mode"] is True
    assert "사진에 보이지 않는 것을 지어내지 마" in PHOTO_SYSTEM


async def test_context_scenes_send_no_image():
    llm = ScriptedLLM(GOOD)

    await SceneCurator(llm).for_context("상황")

    assert llm.calls[0]["image_url"] is None


async def test_retries_once_when_the_first_answer_is_broken():
    llm = ScriptedLLM("깨진 응답", GOOD)

    scenes = await SceneCurator(llm).for_context("상황")

    assert len(scenes) == 3
    assert len(llm.calls) == 2


async def test_gives_up_after_two_broken_answers():
    llm = ScriptedLLM("깨진", "또 깨진")

    with pytest.raises(LLMError):
        await SceneCurator(llm).for_context("상황")


async def test_empty_situation_is_an_error_without_calling_the_llm():
    llm = ScriptedLLM()

    with pytest.raises(LLMError):
        await SceneCurator(llm).for_photo("")
    with pytest.raises(LLMError):
        await SceneCurator(llm).for_context("  ")
    assert llm.calls == []
