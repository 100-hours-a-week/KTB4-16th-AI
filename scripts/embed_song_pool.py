"""곡 풀(song_pool.json)의 설명 문장을 KURE-v1로 임베딩해서 각 곡에 description_embedding을 붙인다.

서버의 질문 벡터와 같은 모델·같은 방식이어야 비교가 된다. 서버(app/components/kure_model.py)는
sentence-transformers 설정대로 첫 토큰(CLS) 벡터를 길이 1로 정규화해 쓰므로 여기서도 똑같이 만든다.
만든 파일은 공개 레포에 넣지 않고 비공개 HF 저장소(dionypark/mulo-song-pool)에 새 태그로 올린다.

실행 (torch·transformers가 있는 임시 환경에서):
    python scripts/embed_song_pool.py <입력 song_pool.json> <출력 song_pool.json>
"""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "nlpai-lab/KURE-v1"
MAX_TOKENS = 512
BATCH = 32
DECIMALS = 5  # 파일 크기를 줄이려고 자른다 — 코사인 값에는 영향이 없는 수준


def embed(texts: list[str]) -> list[list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = tokenizer(
            texts[i : i + BATCH],
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            cls = model(**batch).last_hidden_state[:, 0]
        vecs = torch.nn.functional.normalize(cls, dim=1)
        out.extend([round(x, DECIMALS) for x in v] for v in vecs.tolist())
        print(f"{min(i + BATCH, len(texts))}/{len(texts)}", flush=True)
    return out


def main(src: str, dst: str) -> None:
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    songs = data["songs"]
    vectors = embed([s["description"] for s in songs])
    for song, vec in zip(songs, vectors, strict=True):
        song["description_embedding"] = vec
    data["embedding_model"] = MODEL_NAME
    Path(dst).write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{dst}: {len(songs)}곡, {len(vectors[0])}차원")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
