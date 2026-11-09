"""사진·짤과 VLM 설명문을 한 화면에 모아 보는 확인용 HTML(data/photo_eval/review.html)을 만든다.

카드 하나에 사진, 이전 프롬프트(D)와 고친 프롬프트(E)의 설명문, 각 설명에 대한 Claude의 판정
(judgments_D.json, judgments_E.json), VLM 태그(C), CLIP 태그가 같이 나온다. 읽는 파일이 없으면
그 칸은 "아직 없음"으로 둔다.

- 공개 사진: data/photo_eval/images (출처는 sources.json), 설명은 speed_D.csv·speed_E.csv
- 내 짤: data/photo_eval/memes, 설명은 speed_memes_D.csv·speed_memes_E.csv
- 태그: speed_C.csv(VLM), clip_compare.csv(CLIP, RunPod 쪽)

사진은 HTML 안에 글자(base64)로 넣어서, 파일 하나만으로 어디서 열어도 사진이 보인다.
맥에서는 sips로 긴 변 640픽셀로 줄여 넣고, sips가 없으면 원본을 그대로 넣는다.
짤이 들어 있으니 이 파일을 저장소에 올리거나 다른 사람에게 보내지 않는다.

실행: .venv/bin/python scripts/photo_eval/build_review.py   (표준 라이브러리만 쓴다)
"""

import base64
import csv
import html
import json
import math
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/photo_eval"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EMBED_SIZE = 640
LEVEL_LABEL = {"없음": "문제 없음", "사소": "사소한 오류", "중대": "중대한 오류"}


def read_rows(name: str) -> list[dict]:
    path = DATA / name
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def is_number(value: str | None) -> bool:
    return bool(value) and value.replace(".", "", 1).isdigit()


def descriptions(*names: str) -> dict[str, dict]:
    """사진 이름(확장자 뺀 것) → {text, seconds}. 성공한 줄만 담는다."""
    found = {}
    for name in names:
        for row in read_rows(name):
            if is_number(row.get("seconds")):
                found[Path(row["photo"]).stem] = {
                    "text": row["answer"].strip(),
                    "seconds": float(row["seconds"]),
                }
    return found


def load_judgments(name: str, items: list[dict]) -> dict[str, dict]:
    """판정 파일에는 눈에 띄는 것만 적혀 있어서, 파일이 있으면 없는 항목은 문제 없음으로 본다."""
    path = DATA / name
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    judged = {k: v for k, v in raw.items() if not k.startswith("_")}
    for item in items:
        judged.setdefault(item["id"], {"level": "없음", "note": ""})
    return judged


def vlm_tags() -> dict[str, str]:
    tags = {}
    for row in read_rows("speed_C.csv"):
        try:
            answer = json.loads(row["answer"])
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        tags[Path(row["photo"]).stem] = " / ".join(f"{k}={v}" for k, v in answer.items() if v)
    return tags


def clip_tags() -> dict[str, str]:
    tags = {}
    for row in read_rows("clip_compare.csv"):
        if row.get("scene_runpod"):
            tags[Path(row["photo"]).stem] = (
                f"{row['scene_runpod']} / {row.get('axes_runpod') or '-'}"
            )
    return tags


def load_items() -> list[dict]:
    """카드로 보여 줄 항목: 공개 사진(sources.json 순서) 뒤에 내 짤."""
    items = []
    sources = DATA / "sources.json"
    if sources.exists():
        for record in json.loads(sources.read_text(encoding="utf-8")):
            items.append(
                {
                    "id": record["id"],
                    "src": record["file"],
                    "label": f"검색 힌트 {record['hint_axis']}:{record['hint']}",
                    "memo": record.get("review", ""),
                }
            )
    memes = DATA / "memes"
    if memes.exists():
        for path in sorted(memes.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                items.append(
                    {"id": path.stem, "src": f"memes/{path.name}", "label": "내 짤", "memo": ""}
                )
    return items


def embed_image(src: str) -> str:
    """사진을 HTML 안에 넣을 data URI로 바꾼다. 가능하면 줄여서 넣는다."""
    path = DATA / src
    if shutil.which("sips"):
        with tempfile.TemporaryDirectory() as tmp:
            small = Path(tmp) / "small.jpg"
            done = subprocess.run(
                [
                    "sips",
                    "-Z",
                    str(EMBED_SIZE),
                    "-s",
                    "format",
                    "jpeg",
                    str(path),
                    "--out",
                    str(small),
                ],
                capture_output=True,
            )
            if done.returncode == 0 and small.exists():
                return "data:image/jpeg;base64," + base64.b64encode(small.read_bytes()).decode()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    k = (len(ordered) - 1) * p / 100
    low, high = math.floor(k), math.ceil(k)
    return ordered[low] + (ordered[high] - ordered[low]) * (k - low)


def time_summary(label: str, found: dict[str, dict]) -> str:
    if not found:
        return f"<li>{label}: 아직 없음</li>"
    seconds = [d["seconds"] for d in found.values()]
    return (
        f"<li>{label}: {len(seconds)}장, 중앙값 {statistics.median(seconds):.2f}초, "
        f"p95 {percentile(seconds, 95):.2f}초</li>"
    )


def judge_summary(label: str, judged: dict[str, dict]) -> str:
    if not judged:
        return f"<li>Claude 판정({label}): 아직 없음</li>"
    counts = {
        level: sum(1 for v in judged.values() if v["level"] == level) for level in LEVEL_LABEL
    }
    text = " · ".join(f"{LEVEL_LABEL[k]} {v}" for k, v in counts.items())
    return f"<li>Claude 판정({label}): {text}</li>"


def description_block(title: str, found: dict | None, verdict: dict | None) -> str:
    if not found:
        return f'<div class="desc"><b>{title}</b><p class="none">아직 없음</p></div>'
    text = html.escape(found["text"])
    badge = ""
    if verdict:
        level = verdict["level"]
        detail = f" — {html.escape(verdict['note'])}" if verdict["note"] else ""
        badge = f'<p class="judge {level}"><b>Claude 판정: {LEVEL_LABEL[level]}</b>{detail}</p>'
    return (
        f'<div class="desc"><b>{title}</b> <small>{len(found["text"])}자, '
        f"{found['seconds']:.1f}초</small><p>{text}</p>{badge}</div>"
    )


def card(item: dict, tables: dict) -> str:
    pid = item["id"]
    memo = f'<p class="memo">{html.escape(item["memo"])}</p>' if item["memo"] else ""
    return f"""<section>
<img src="{item["uri"]}" alt="{pid}">
<div class="body">
<h3>{pid} <small>{html.escape(item["label"])}</small></h3>
{description_block("D 이전 프롬프트", tables["old"].get(pid), tables["judged_d"].get(pid))}
{description_block("E 고친 프롬프트", tables["new"].get(pid), tables["judged_e"].get(pid))}
<p class="tags"><b>VLM 태그(C)</b> {html.escape(tables["c_tags"].get(pid, "-"))}</p>
<p class="tags"><b>CLIP 태그</b> {html.escape(tables["clip"].get(pid, "-"))}</p>
{memo}</div></section>"""


STYLE = """
body{font-family:-apple-system,"Apple SD Gothic Neo",sans-serif;max-width:1040px;margin:24px auto;
padding:0 16px;color:#222}
section{display:flex;gap:16px;margin:0 0 18px;padding:12px;border:1px solid #ddd;border-radius:8px}
img{width:300px;max-width:38%;height:auto;object-fit:contain;align-self:flex-start;border-radius:4px}
.body{flex:1;min-width:0}h3{margin:0 0 6px}small{font-weight:normal;color:#777}
.desc{margin:8px 0}.desc p{margin:4px 0;line-height:1.55}.none{color:#999}
.judge{margin:4px 0;padding:6px 8px;border-radius:4px;font-size:13px}
.judge.없음{background:#e8f5e9}.judge.사소{background:#fff8e1}.judge.중대{background:#ffebee}
.tags{margin:3px 0;font-size:13px;color:#444}.memo{color:#b45309;font-size:12px}
"""


def main() -> None:
    items = load_items()
    for item in items:
        item["uri"] = embed_image(item["src"])
    tables = {
        "old": descriptions("speed_D.csv", "speed_memes_D.csv"),
        "new": descriptions("speed_E.csv", "speed_memes_E.csv"),
        "judged_d": load_judgments("judgments_D.json", items),
        "judged_e": load_judgments("judgments_E.json", items),
        "c_tags": vlm_tags(),
        "clip": clip_tags(),
    }
    memes = sum(1 for item in items if item["label"] == "내 짤")
    cards = "\n".join(card(item, tables) for item in items)
    page = f"""<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>사진·짤 VLM 설명문 확인</title><style>{STYLE}</style>
<h2>사진 {len(items) - memes}장 + 짤 {memes}장</h2>
<ul>{time_summary("D 이전 프롬프트", tables["old"])}{time_summary("E 고친 프롬프트", tables["new"])}
{judge_summary("D", tables["judged_d"])}{judge_summary("E", tables["judged_e"])}</ul>
<p>판정 파일에는 눈에 띄는 항목만 적고 나머지는 문제 없음으로 센 것이며, 틀릴 수 있습니다.
설명이 사진과 맞는지 직접 확인하세요.</p>
{cards}</html>"""
    out = DATA / "review.html"
    out.write_text(page, encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    counts = f"D {len(tables['old'])}개, E {len(tables['new'])}개"
    print(f"{out} ({len(items)}개 카드, {counts}, {size_mb:.1f}MB)")


if __name__ == "__main__":
    main()
