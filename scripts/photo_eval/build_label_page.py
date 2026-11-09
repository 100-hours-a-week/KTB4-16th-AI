"""정답 초안을 사람이 확인하고 고치는 화면(data/photo_eval/label.html)을 만든다.

사진마다 속성(장소·사진 종류·대상·분위기·표정·감정·노래 분위기)을 체크박스로 고르고, 다 확인하면
"정답 파일 내보내기"로 gold_labels.json을 받는다. 이미 gold_labels.json이 있으면 그 값으로,
없으면 Claude가 쓴 초안(gold_labels_draft.json)으로 채워서 연다.

사진은 HTML 안에 글자(base64)로 넣어서 파일 하나만으로 열린다. 짤이 들어 있으니 이 파일을
저장소에 올리거나 다른 사람에게 보내지 않는다.

실행: PYTHONPATH=. .venv/bin/python scripts/photo_eval/build_label_page.py
"""

import json
from pathlib import Path

from scripts.photo_eval.build_review import embed_image, load_items
from scripts.photo_eval.gold import (
    ATTRS,
    DATA,
    DRAFT_PATH,
    GOLD_PATH,
    LIMITS,
    load_gold,
    load_vocab,
)

TEMPLATE = Path(__file__).with_name("label_template.html")
NOTE = (
    "사람이 확인한 사진 정답. confirmed가 true면 사람이 확인한 것이다. 장소가 비어 있으면 "
    "'목록에 맞는 곳 없음'이다. 그림·짤은 대상·분위기를, 진짜 사진은 표정·감정을 비운다."
)


def main() -> None:
    source = GOLD_PATH if GOLD_PATH.exists() else DRAFT_PATH
    labels = load_gold(source)
    items = load_items()
    missing = [item["id"] for item in items if item["id"] not in labels]
    if missing:
        raise SystemExit(f"{source.name}에 없는 사진이 있어요: {missing}")
    for item in items:
        item["uri"] = embed_image(item["src"])
    data = {
        "items": [{"id": i["id"], "label": i["label"], "uri": i["uri"]} for i in items],
        "labels": {i["id"]: labels[i["id"]] for i in items},
        "vocab": load_vocab(),
        "limits": LIMITS,
        "attrs": ATTRS,
        "note": NOTE,
    }
    page = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, ensure_ascii=False)
    )
    out = DATA / "label.html"
    out.write_text(page, encoding="utf-8")
    confirmed = sum(1 for v in data["labels"].values() if v.get("confirmed"))
    print(f"{out} ({len(items)}개 카드, 확인됨 {confirmed}개, 읽은 파일 {source.name})")


if __name__ == "__main__":
    main()
