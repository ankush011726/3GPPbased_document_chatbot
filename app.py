"""
Chat interface. Run with:
    streamlit run app.py
"""

import os

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

import config
from generator import answer_question, get_llm, rewrite_query
from ingest import load_vector_store
from retriever import Retriever

st.set_page_config(page_title="RAG Chat", page_icon="📚")
st.title("📚 Ask your documents")

# --- one-time setup ---
if "ready" not in st.session_state:
    if not os.path.exists(config.CHROMA_DIR):
        st.error("No vector store found. Add PDFs to docs/ and run `python ingest.py` first.")
        st.stop()

    db = load_vector_store()
    st.session_state.retriever = Retriever(db)
    st.session_state.llm = get_llm()
    st.session_state.messages = []   # what the user sees
    st.session_state.history = []    # what the LLM uses for context
    st.session_state.ready = True

# --- show past messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- new question ---
if question := st.chat_input("Ask a question..."):
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. rewrite follow-ups so they can be searched alone
            search_query = rewrite_query(question, st.session_state.history, st.session_state.llm)
            # 2. retrieve (multi-query + hybrid + RRF + rerank)
            chunks = st.session_state.retriever.retrieve(search_query)
            # 3. answer with Groq
            response = answer_question(question, chunks, st.session_state.llm, st.session_state.history)

        st.markdown(response)

        # show what was retrieved so you can debug bad answers
        with st.expander("📄 Retrieved chunks"):
            if chunks:
                for i, c in enumerate(chunks, 1):
                    source = c.metadata.get("source", "unknown")
                    page = c.metadata.get("page")
                    label = f"{source}, p. {page}" if page else source
                    st.markdown(f"**{i}. {label}**")
                    st.markdown(c.page_content)
            else:
                st.markdown("Nothing retrieved.")

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.history.append({"user": question, "assistant": response})
