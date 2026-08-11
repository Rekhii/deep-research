import argparse

from src.agent import build_graph
from src.retriever import close_client


def run(question: str) -> dict:
    """
    Runs one question through the graph and returns the final state.

    Every key the State schema declares must be present at entry.
    LangGraph merges partial dicts from nodes but does not invent keys,
    so a node reading state["draft"] before write_node has run would
    raise KeyError without these defaults.
    """
    graph = build_graph()

    initial = {
        "question": question,
        "query": "",
        "chunks": [],
        "draft": "",
        "verdict": "",
        "critique": "",
        "retrieval_attempts": 0,
        "revision_attempts": 0,
    }

    return graph.invoke(initial)


def main():
    parser = argparse.ArgumentParser(description="Ask a question about the ingested papers.")
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--trace", action="store_true", help="Show loop counters and sources")
    args = parser.parse_args()

    try:
        final = run(args.question)

        print()
        print(final["draft"])

        if args.trace:
            # The numbered sources match the [n] citations in the answer,
            # because format_chunks numbers them in this same order.
            print()
            print("SOURCES")
            for i, c in enumerate(final["chunks"], start=1):
                p = c["payload"]
                print(f"  [{i}] {p['section']}  (chunk {p['chunk_index']})")

            print()
            print(f"query used:         {final['query']}")
            print(f"retrieval attempts: {final['retrieval_attempts']}")
            print(f"revision attempts:  {final['revision_attempts']}")
    except ConnectionError:
        # ollama.chat raises this when the daemon is not reachable. Without
        # catching it the user gets 25 lines of LangGraph internals ending in
        # a message that looks like this project is broken, when the actual
        # problem is a missing prerequisite.
        print()
        print("Could not reach Ollama.")
        print()
        print("  1. Start it:   ollama serve")
        print("  2. Check the model is pulled:   ollama list")
        print("     If qwen3:8b is missing:      ollama pull qwen3:8b")
        raise SystemExit(1)

    finally:
        # Embedded Qdrant is single-writer and holds a lock on the folder.
        # Without this, an unclean exit leaves the lock held and the next
        # run fails with a storage-already-accessed error.
        close_client()


if __name__ == "__main__":
    main()