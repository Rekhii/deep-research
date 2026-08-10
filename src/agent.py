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



# The rewriter only ever sees the question and the last verdict.
# Keeping the instruction in a system message separates "who you are"
# from "what to do right now", which qwen3 follows more reliably.
REWRITE_SYSTEM = """You rewrite search queries for a document retrieval system.
Output ONLY the rewritten query. No explanation, no quotes, no preamble."""


def rewrite_node(state: State) -> dict:
    """
    Turns the user's question into a search query.

    First pass: use the question as-is, no LLM call.
    Later passes: rewrite, because the previous query failed to retrieve enough.
    """
    attempts = state["retrieval_attempts"]

    # Attempt 0 is the happy path. The user's own wording is usually the best
    # query we have, and burning an LLM call to paraphrase it before we know
    # anything is wrong just adds latency and a chance to lose meaning.
    if attempts == 0:
        return {
            "query": state["question"],
            "retrieval_attempts": 1,
        }

    # We only reach here because the grader said the last retrieval was thin.
    # So we tell the model what already failed and ask for a different angle.
    prompt = f"""Original question: {state["question"]}

Previous search query: {state["query"]}
That query did not retrieve enough relevant material.

Write a different search query for the same question. Use alternative
terminology, synonyms, or more specific technical vocabulary that would
appear in an academic paper on this topic."""

    new_query = llm(prompt, system=REWRITE_SYSTEM)

    return {
        "query": new_query,
        "retrieval_attempts": attempts + 1,
    }                                             # Only this key changes; LangGraph merges it into state



def retrieve_node(state: State):
    '''
    Runs the hybrid retrieval pipeline on the current query.
    Also increments the attempt counter that caps the correction loop.
    '''

    chunks = retrieve(state["query"])                                           # dense + sparse + RRF + rerank, returns TOP_K chunks

    attempts = state.get("retrieval_attempts", 0) + 1                           # Count this attempt; the router reads this to cap the loop

    return {
        "chunks": chunks,                                                       # Overwritten each pass, not accumulated
        "retrieval_attempts": attempts,
    }


























