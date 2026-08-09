from typing import TypedDict, List, Dict, Any

import ollama
from langgraph.graph import StateGraph, END

from src.retriever import retrieve, close_client
from src.config import LLM_MODEL, NUM_CTX


class State(TypedDict):
    '''
    The shared object every node reads from and writes to.
    LangGraph passes this dict from node to node. A node returns only the keys
    it wants to change, and LangGraph merges those back into the full state.
    '''

    question: str                   # the original user question, never modified
    query: str                      # the current search query, rewritten by the loop
    chunks: List[Dict[str, Any]]    # retrieved chunks from retriever.retrieve
    draft: str                      # the current answer draft
    verdict: str                    # grader output: "sufficient" or "insufficient"
    critique: str                   # critic feedback on the draft
    retrieval_attempts: int
    revision_attempts: int


def llm(prompt, system=None, think=False):
    '''
    The single place where the project talks to the language model.
    Every node calls this instead of calling ollama.chat directly.
    '''

    messages = []                                        # Ollama wants a list of role/content dicts

    if system:                                           # System prompt is optional, some nodes do not need one
        messages.append({
            "role": "system",                            # Sets the model's role or rules for this call
            "content": system,
        })

    messages.append({
        "role": "user",                                  # The actual task text
        "content": prompt,
    })

    response = ollama.chat(
        model=LLM_MODEL,                                 # qwen3:8b, the one model used by every node
        messages=messages,
        think=think,                                     # False strips the reasoning trace out of the output
        options={"num_ctx": NUM_CTX},                    # 8192. Must be explicit or Ollama silently uses a smaller window.
    )

    return response["message"]["content"].strip()        # Dig the text out of the response and trim whitespace



REWRITE_SYSTEM = (                                       # Module-level constant, sits next to the node that uses it
    "You rewrite questions into short search queries for a research paper "
    "database. Output only the query. No explanation, no punctuation at the end."
)


def rewrite_node(state: State):
    '''
    Turns the user's question into a search query.
    First pass compresses the question. Later passes rewrite it differently
    because the grader rejected what the previous query found.
    '''

    attempt = state.get("retrieval_attempts", 0)            # 0 on first pass; .get avoids KeyError before the key exists

    if attempt == 0:                                        # Nothing has failed yet, just compress
        prompt = (
            f"Question: {state['question']}\n\n"
            "Write a short search query, 3 to 8 words, capturing the key terms."
        )
    else:                                                   # Previous attempt was graded insufficient
        prompt = (                                          # Feed the failed query in so the model does not repeat it
            f"Question: {state['question']}\n"
            f"Previous query that found nothing useful: {state['query']}\n\n"
            "Write a different search query, 3 to 8 words. Use different wording "
            "or broader terms than the previous query."
        )

    new_query = llm(prompt, system=REWRITE_SYSTEM)      # think=False by default, we want the bare query

    return {"query": new_query}                         # Only this key changes; LangGraph merges it into state



def retrieve_node(state: State):
    '''
    Runs the hybrid retrieval pipeline on the current query.
    Also increments the attempt counter that caps the correction loop.
    '''

    chunks = retrieve(state["query"])                       # dense + sparse + RRF + rerank, returns TOP_K chunks

    attempts = state.get("retrieval_attempts", 0) + 1       # Count this attempt; the router reads this to cap the loop

    return {
        "chunks": chunks,                                   # Overwritten each pass, not accumulated
        "retrieval_attempts": attempts,
    }


























