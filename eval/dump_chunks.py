"""
eval/dump_chunks.py

Reads every point out of the Qdrant collection and prints it with its
chunk_index plus a short text preview, sorted into document order. The
output is scratch material for hand-building eval/golden.jsonl: you read
through it, decide which chunks actually answer each question, and copy
those chunk_index numbers into the golden file.

Run from the repo root:
    python -m eval.dump_chunks | Out-File -Encoding utf8 eval\chunks.txt
"""

import sys

from qdrant_client import QdrantClient

# COLLECTION may be named differently in your config (COLLECTION_NAME etc).
# Check src/config.py and adjust this import if the name doesn't match.
from src.config import QDRANT_PATH, COLLECTION

PREVIEW_CHARS = 220   # how much of each chunk to show; enough to judge relevance
BATCH = 64            # how many points to pull per scroll call


def main():
    # PowerShell's default console encoding is cp1252, which cannot represent
    # Greek letters, math symbols, or smart quotes that appear in the paper.
    # Force UTF-8 on stdout; errors="replace" swaps anything still unmappable
    # for "?" instead of raising UnicodeEncodeError mid-dump.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Embedded Qdrant takes a filesystem lock. Make sure app.py or any other
    # process holding the DB is closed, or this call will fail.
    client = QdrantClient(path=QDRANT_PATH)

    offset = None   # scroll cursor; Qdrant hands back the next one each call
    rows = []       # collect everything first so it can be sorted before printing

    while True:
        # scroll() walks the whole collection without needing a query vector.
        # with_vectors=False keeps 768 floats per point out of memory.
        points, offset = client.scroll(
            collection_name=COLLECTION,
            limit=BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for p in points:
            payload = p.payload or {}

            # Print the payload schema once, as a sanity check that the field
            # names here still match what ingest.py writes.
            if not rows:
                print(f"# payload keys: {list(payload.keys())}\n")

            # chunk_index is the label the golden file keys on: a short
            # integer, unique per document, far less error-prone to hand-copy
            # than a 36-character uuid.
            rows.append((
                payload.get("chunk_index"),
                payload.get("section", "?"),
                # Collapse newlines and tabs so each preview stays on one line.
                " ".join(payload.get("text", "").split())[:PREVIEW_CHARS],
            ))

        # Qdrant returns offset=None once there are no more pages.
        if offset is None:
            break

    # Sort by chunk_index so the dump reads in document order rather than
    # Qdrant's internal scroll order. Chunks that answer the same question
    # tend to sit next to each other in the paper, which makes hand-labeling
    # far less painful when they are adjacent in this file too.
    rows.sort(key=lambda r: r[0])

    for chunk_index, section, preview in rows:
        print(f"chunk_index={chunk_index}  section={section}")
        print(f"    {preview}")
        print()

    print(f"# total: {len(rows)} chunks")


if __name__ == "__main__":
    main()