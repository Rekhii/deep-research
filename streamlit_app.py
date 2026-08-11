"""
streamlit_app.py

Browser UI over the same graph app.py uses. Kept deliberately thin: all it does
is call run() and display the result, so there is no second code path that
could drift from the CLI.

Run:
    streamlit run streamlit_app.py
"""

import streamlit as st

from app import run

st.set_page_config(page_title="deep-research", layout="centered")

st.title("deep-research")
st.caption("Local question answering over ingested research papers. "
           "Hybrid retrieval, self-correcting agent, no external APIs.")

question = st.text_input(
    "Question",
    placeholder="What are the three phases of the LIDA cognitive cycle?",
)

# Streamlit reruns this whole script on every interaction, so the work has to
# be gated behind an explicit button rather than firing on each keystroke.
if st.button("Ask", type="primary") and question.strip():
    try:
        # A query takes roughly 13s: retrieval, grading, drafting and critique
        # each make an LLM call, and the loops may run twice.
        with st.spinner("Retrieving, drafting and checking the answer..."):
            final = run(question)

    except ConnectionError:
        st.error(
            "Could not reach Ollama. Start it with `ollama serve`, "
            "and make sure `qwen3:8b` is pulled."
        )

    else:
        st.markdown(final["draft"])

        st.divider()

        # The numbered sources line up with the [n] citations in the answer,
        # because format_chunks numbers them in this same order.
        st.subheader("Sources")
        for i, c in enumerate(final["chunks"], start=1):
            p = c["payload"]
            with st.expander(f"[{i}] {p['section']}  (chunk {p['chunk_index']})"):
                st.write(p["text"])

        # Exposing the loop counters makes the agent's self-correction visible.
        # A question the grader rejected on the first pass shows 2 retrieval
        # attempts and a rewritten query, which is the interesting case.
        with st.expander("Trace"):
            st.write(f"**Query used:** {final['query']}")
            st.write(f"**Retrieval attempts:** {final['retrieval_attempts']}")
            st.write(f"**Revision attempts:** {final['revision_attempts']}")