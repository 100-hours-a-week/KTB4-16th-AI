"""백엔드 역할 흉내: 더미 자물쇠를 /api/embeddings/generate로 보낸다.

사용법:
    python scripts/send_dummy_locks.py                       # fixtures/dummy_records.json
    python scripts/send_dummy_locks.py 다른파일.json
"""

import json
import os
import sys
from pathlib import Path

import httpx

AI_URL = os.getenv("AI_URL", "http://localhost:8000")
TOKEN = os.getenv("INTERNAL_TOKEN", "dev-internal-token")


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures/dummy_records.json")
    records = json.loads(path.read_text(encoding="utf-8"))

    with httpx.Client(base_url=AI_URL, headers={"X-Internal-Token": TOKEN}) as client:
        for record in records:
            res = client.post("/api/embeddings/generate", json=record)
            print(record["recordId"], res.status_code, res.text)


if __name__ == "__main__":
    main()
