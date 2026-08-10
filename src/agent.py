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



def retrieve_node(state: State) -> dict:
    """
    Runs hybrid retrieval for the current query.

    All the real work (dense + sparse + RRF + rerank) lives in retriever.py,
    which already returns plain dicts. This node just parks them in state.
    """
    return {"chunks": retrieve(state["query"])}




def format_chunks(chunks: list) -> str:
    """
    Turns the chunk list into numbered context for a prompt.

    Numbering matters: the writer and critic both need a way to point at a
    specific chunk, and "[3]" is something an 8B model produces reliably
    where a chunk id or a quoted phrase is not.
    """
    parts = []
    for i, c in enumerate(chunks, start=1):
        p = c["payload"]
        parts.append(f"[{i}] (from {p['section']})\n{p['text']}")
    return "\n\n".join(parts)





GRADE_SYSTEM = """You judge whether retrieved text can answer a question.
Answer with exactly one word: YES or NO. Nothing else."""


def grade_node(state: State) -> dict:
    """
    Decides whether the retrieved chunks are enough to answer the question.

    Each chunk is graded independently with a binary YES/NO. We count the
    YES votes rather than asking the model to judge the whole set at once.
    """
    question = state["question"]
    chunks = state["chunks"]

    relevant = 0
    for c in chunks:
        prompt = f"""Question: {question}

Retrieved text:
{c["payload"]["text"]}

Does this text contain information useful for answering the question?"""

        answer = llm(prompt, system=GRADE_SYSTEM).strip().upper()

        # Substring check, not equality. The model occasionally emits
        # "YES." or "YES, it does" despite the instruction, and treating
        # those as NO would silently poison the verdict.
        if answer.startswith("Y"):
            relevant += 1

    # One good chunk is enough to write from. Demanding more sends the loop
    # off rewriting queries that were actually working.
    verdict = "sufficient" if relevant >= 1 else "insufficient"

    return {"verdict": verdict}



WRITE_SYSTEM = """You answer questions using only the provided context.

Rules:
- Use only information from the numbered context passages.
- Cite every claim with the passage number in square brackets, like [2].
- If the context does not support an answer, say so plainly.
- Do not use outside knowledge, even if you are confident it is correct."""


def write_node(state: State) -> dict:
    """
    Writes the answer from the retrieved chunks.

    Handles both the first draft and revisions. If a critique exists in
    state, this is a revision pass and the feedback goes into the prompt.
    """
    context = format_chunks(state["chunks"])
    critique = state.get("critique", "")

    if critique:
        # Revision pass. The model sees its own previous draft and what
        # was wrong with it, so it can fix specific problems rather than
        # starting over and possibly losing what was already correct.
        prompt = f"""Question: {state["question"]}

Context passages:
{context}

Your previous draft:
{state["draft"]}

Problems found with that draft:
{critique}

Rewrite the answer, fixing those problems. Keep what was correct."""
    else:
        # First draft.
        prompt = f"""Question: {state["question"]}

Context passages:
{context}

Write a clear, well-organised answer to the question."""

    draft = llm(prompt, system=WRITE_SYSTEM)

    return {"draft": draft}




# Critique

CLAIM_SYSTEM = """You extract factual claims from text.
Output one claim per line. No numbering, no bullets, no commentary.
Each line must be a single standalone factual statement."""

CHECK_SYSTEM = """You verify whether a claim is supported by context.
Answer with exactly one word: YES or NO. Nothing else."""


def critique_node(state: State) -> dict:
    """
    Checks the draft's claims against the retrieved context.

    Two stages: split the draft into individual claims, then verify each
    one separately. Unsupported claims become the critique that the write
    node uses on its revision pass.
    """
    draft = state["draft"]
    context = format_chunks(state["chunks"])
    attempts = state.get("revision_attempts", 0)

    # Stage 1: decompose the draft into checkable statements.
    claims_raw = llm(
        f"Extract the factual claims from this text:\n\n{draft}",
        system=CLAIM_SYSTEM,
    )
    claims = [c.strip() for c in claims_raw.split("\n") if c.strip()]

    # Stage 2: verify each claim independently against the context.
    unsupported = []
    for claim in claims:
        prompt = f"""Context:
{context}

Claim: {claim}

Is this claim supported by the context above?"""

        answer = llm(prompt, system=CHECK_SYSTEM).strip().upper()
        if not answer.startswith("Y"):
            unsupported.append(claim)

    # No problems found means the draft is done and the loop exits.
    if not unsupported:
        return {"critique": "", "revision_attempts": attempts + 1}

    critique = "These claims are not supported by the context:\n" + "\n".join(
        f"- {c}" for c in unsupported
    )

    return {"critique": critique, "revision_attempts": attempts + 1}


def route_after_grade(state: State) -> str:
    """
    Retrieval correction loop. Decides whether to answer or search again.

    Returns a node name. LangGraph calls this after grade_node and sends
    control wherever the string points.
    """
    # Cap first, before looking at the verdict. Two failed retrievals means
    # the query is not the problem and rewriting again just burns time.
    # Better to answer honestly from weak context than loop forever.
    if state["retrieval_attempts"] >= 2:
        return "write"

    if state["verdict"] == "insufficient":
        return "rewrite"

    return "write"


def route_after_critique(state: State) -> str:
    """
    Revision loop. Decides whether to revise the draft or finish.
    """
    # Same reasoning: cap before content. Two revisions is where returns
    # stop and the model starts rewording rather than fixing.
    if state["revision_attempts"] >= 2:
        return END

    # Empty critique means the critic found nothing wrong.
    if not state["critique"]:
        return END

    return "write"


def build_graph():
    """
    Wires the five nodes into the two-loop graph.
    """
    g = StateGraph(State)

    g.add_node("rewrite", rewrite_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("write", write_node)
    g.add_node("critique", critique_node)

    g.set_entry_point("rewrite")

    # Linear spine: every query passes through these in order.
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")

    # Retrieval correction loop: back to rewrite, or forward to write.
    g.add_conditional_edges("grade", route_after_grade, {
        "rewrite": "rewrite",
        "write": "write",
    })

    g.add_edge("write", "critique")

    # Revision loop: back to write, or done.
    g.add_conditional_edges("critique", route_after_critique, {
        "write": "write",
        END: END,
    })

    return g.compile()




