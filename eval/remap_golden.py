import sys

from qdrant_client import QdrantClient

from src.config import QDRANT_PATH, COLLECTION

# One distinctive phrase per question, taken from the gold chunk's text. These
# were chosen to be specific enough that they appear in exactly one chunk. A
# phrase matching several chunks usually means the answer straddles a boundary,
# which is worth knowing too.
PROBES = {
    "q01": "acronym for Learning Intelligent Distribution Agent",
    "q02": "must frequently and continually sense its environment",
    "q03": "largely based on Baars",
    "q04": "Six such principles are discussed below",
    "q05": "with the exception of two serial decision points",
    "q06": "an understanding phase, an attending phase",
    "q07": "small, special-purpose processors",
    "q08": "coalition containing the most salient",
    "q09": "consciously mediated action selection",
    "q10": "Baars incorporated ideomotor theory",
    "q11": "planning, scheduling, and problem solving",
    "q12": "perceptual , episodic , and procedural",
    "q13": "can only occur after information has been attended to",
    "q14": "feelings with cognitive content",
    "q15": "performed by appraisal codelets",
    "q16": "tentatively assigned neural correlates",
    "q17": "assumes a strictly modular organization of the brain",
    "q18": "Observer design pattern",
    "q19": "283 ms averaged over 30 runs",
    "q20": "Workspace serving as the blackboard",
}


def load_chunks(client):
    """Pull every chunk into a list of (chunk_index, section, text)."""
    rows = []
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
            rows.append((
                payload.get("chunk_index"),
                payload.get("section", "?"),
                payload.get("text", ""),
            ))
        if offset is None:
            break
    rows.sort(key=lambda r: r[0])
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    client = QdrantClient(path=QDRANT_PATH)
    rows = load_chunks(client)
    print(f"{len(rows)} chunks in collection\n")

    for qid, probe in PROBES.items():
        # A phrase can legitimately land in more than one chunk because of the
        # 100-character overlap between neighbours. When that happens both
        # indices are usually valid gold, so print all matches.
        hits = [(idx, section) for idx, section, text in rows if probe in text]

        if not hits:
            print(f"{qid}  NO MATCH for {probe!r}")
            print("      probe phrase may have been altered by clean(); search by hand")
        else:
            found = ", ".join(str(i) for i, _ in hits)
            print(f"{qid}  -> [{found}]   ({hits[0][1]})")
        print()


if __name__ == "__main__":
    main()