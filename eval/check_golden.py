"""
eval/check_golden.py

Prints every question in golden.jsonl alongside the FULL text of each chunk
it labels as gold. Read each one and confirm the chunk really contains the
answer, not merely related discussion. A wrong gold label makes the harness
report a low score for a retrieval that actually worked.

Run:
    python -m eval.check_golden | Out-File -Encoding utf8 eval\check.txt
"""

import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from src.config import QDRANT_PATH, COLLECTION

GOLDEN = Path("eval/golden.jsonl")


def load_chunks(client):
    """Pull every chunk into a dict keyed by chunk_index for fast lookup."""
    chunks = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=64,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            chunks[payload.get("chunk_index")] = (
                payload.get("section", "?"),
                payload.get("text", ""),
            )
        if offset is None:
            break
    return chunks


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    client = QdrantClient(path=QDRANT_PATH)
    chunks = load_chunks(client)

    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)

        print("=" * 78)
        print(f"{item['qid']}  [{item.get('difficulty', '?')}]")
        print(f"Q: {item['question']}")
        print()

        for cid in item["gold_ids"]:
            # A missing id means the golden file points at a chunk that no
            # longer exists, e.g. after a re-ingest shifted the boundaries.
            if cid not in chunks:
                print(f"  !! chunk_index={cid} NOT FOUND in collection")
                continue

            section, text = chunks[cid]
            print(f"  --- chunk_index={cid}  section={section}")
            for para in text.split("\n"):
                if para.strip():
                    print(f"      {para.strip()}")
            print()

    print("=" * 78)
    print(f"# {len(chunks)} chunks in collection")


if __name__ == "__main__":
    main()