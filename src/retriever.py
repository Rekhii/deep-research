from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from src.config import (
    QDRANT_PATH,
    COLLECTION,
    DENSE_MODEL,
    SPARSE_MODEL,
    CANDIDATES,
    RERANK_MODEL,
    RRF_K,
    TOP_K,
)

_client = None                                                          # will later store the QdrantClient
_dense = None                                                           # will later store the dense embedding model
_sparse = None                                                          # will later store the sparse embedding model
_reranker = None

def get_client():
    """
    Opens the embedded Qdrant database, reusing a single connection.
    """

    global _client                                                     # Without global Python would treat _client as a new local variable inside the function.

    if _client is None:
        _client = QdrantClient(path = QDRANT_PATH)

    return _client


def get_models():
    """
    Loads the dense and sparse embedding models once and reuses them.
    """
    global _dense, _sparse
    if _dense is None:
        _dense = TextEmbedding(DENSE_MODEL)
    if _sparse is None:
        _sparse = SparseTextEmbedding(SPARSE_MODEL)
    return _dense, _sparse


def close_client():
    """
    Closes the Qdrant connection and releases the folder lock.
    """
    global _client
    if _client is not None:
        _client.close()
        _client = None


def dense_search(query, limit=CANDIDATES):
    """Embeds the query and searches the dense vector index."""

    dense_model, _ = get_models()                                   # Get dense model, ignore sparse model

    vector = list(dense_model.query_embed(query))[0]                # Convert query text into dense vector

    hits = get_client().query_points(
        collection_name=COLLECTION,                                 # Qdrant collection to search
        query=vector.tolist(),                                      # Query vector converted into Python list
        using="dense",                                              # Use dense vector index
        limit=limit,                                                # Number of results to return
        with_payload=True,                                          # Return metadata/text with results
    ).points                                                        # Get only the matched points

    return hits                                                     # Return search results


def sparse_search(query, limit=CANDIDATES):
    """Embeds the query as a sparse BM25 vector and searches the sparse index."""

    _, sparse_model = get_models()                                  # Get sparse model, ignore dense model

    sv = list(sparse_model.query_embed(query))[0]                   # Convert query text into sparse vector

    hits = get_client().query_points(
        collection_name=COLLECTION,                                 # Qdrant collection to search

        query=models.SparseVector(                                  # Create sparse vector in Qdrant format
            indices=sv.indices.tolist(),                            # Positions of important words/features
            values=sv.values.tolist(),                              # Weights/scores of those words/features
        ),

        using="sparse",                                             # Search using sparse index
        limit=limit,                                                # Number of results to return
        with_payload=True,                                          # Also return stored text/metadata
    ).points                                                        # Get the matched results

    return hits                                                     # Return search results



# Reciprocal Rank Fusion
def rrf_fuse(rankings, k=RRF_K, limit=CANDIDATES):                     # 60 in RRF is a constant that reduces how much difference there is between nearby ranks.
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).
    Only the position/rank of a result matters, not its original score.

    Why k=60. It comes from the original 2009 Cormack paper and has survived as
    the default because it works. Its job is to flatten the curve near the top.
    Without it, rank 1 scores 1.0 and rank 2 scores 0.5, so the first list to have an opinion dominates.
    With k=60, rank 1 gives 0.0164 and rank 2 gives 0.0161, a difference small enough that a document
    appearing at rank 3 in both lists beats a document appearing at rank 1 in only one.
    That is the property you want: agreement across retrievers outranks confidence within one.
    """

    scores = {}                                                     # Store final RRF score for each result ID
    payloads = {}                                                   # Store payload/metadata for each result ID

    for hits in rankings:                                           # Go through each ranked result list
        for rank, hit in enumerate(hits, start=1):                  # Go through each result with rank 1, 2, 3...

            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
                                                                    # Add RRF score for this result
                                                                    # Higher-ranked results get more score

            payloads[hit.id] = hit.payload                          # Save the result's payload/metadata

    fused = sorted(
        scores.items(),                                             # Convert scores dict into (id, score) pairs
        key=lambda kv: kv[1],                                       # Sort using the score value
        reverse=True                                                # Highest score comes first
    )

    return [
        {
            "id": pid,                                              # Result ID
            "score": score,                                         # Final combined RRF score
            "payload": payloads[pid]                                # Payload/metadata for that result
        }
        for pid, score in fused[:limit]                             # Return only top 'limit' results
    ]


def get_reranker():
    """
    Loads the cross-encoder once and reuses it.
    """
    global _reranker
    if _reranker is None:
        _reranker = TextCrossEncoder(RERANK_MODEL)
    return _reranker


def rerank(query, candidates, limit=TOP_K):
    """
    Rescore fused candidates using the reranker model
    and return only the best results.
    """

    if not candidates:                                      # If candidate list is empty
        return []                                           # Return empty list

    model = get_reranker()                                  # Load/get the reranking model

    texts = [c["payload"]["text"] for c in candidates]      # Extract only text from each candidate

    scores = list(model.rerank(query, texts))               # Compare query with every candidate text

    for candidate, score in zip(candidates, scores):        # Match each candidate with its new score
        candidate["rerank_score"] = float(score)            # Store reranker score inside candidate

    ranked = sorted(
        candidates,
        key=lambda c: c["rerank_score"],                    # Sort using reranker score
        reverse=True                                        # Highest score first
    )

    return ranked[:limit]                                   # Return only top results



def retrieve(query, candidates=CANDIDATES, top_k=TOP_K):
    """
    Full retrieval pipeline: hybrid search, RRF fusion, cross-encoder rerank.
    Returns the top_k most relevant chunks.

    This is the public face of the file.
    Everything else becomes an implementation detail that agent.py never has to know about.
    """
    dense_hits = dense_search(query, limit=candidates)
    sparse_hits = sparse_search(query, limit=candidates)

    fused = rrf_fuse([dense_hits, sparse_hits], limit=candidates)

    return rerank(query, fused, limit=top_k)