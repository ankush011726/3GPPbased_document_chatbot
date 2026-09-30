"""
Answer generation with Groq, plus query rewriting for multi-turn chat.
"""

from langchain_google_genai import ChatGoogleGenerativeAI

import config


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(model=config.CHAT_MODEL, temperature=0)


def rewrite_query(query: str, history: list[dict], llm: ChatGoogleGenerativeAI) -> str:
    """Make a follow-up question standalone so it can be searched on its own.

    Example: "What about him?" -> "What is the CEO's name?" (using history)
    """
    if not history:
        return query

    history_text = "\n".join(
        f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history
    )
    prompt = (
        "Given the chat history below, rewrite the new question so it is standalone "
        "and can be searched on its own. Return only the rewritten question.\n\n"
        f"Chat history:\n{history_text}\n\nNew question: {query}"
    )
    return llm.invoke(prompt).content.strip()


def answer_question(query: str, chunks: list, llm: ChatGoogleGenerativeAI, history: list[dict] | None = None) -> str:
    """Answer using only the retrieved chunks. Refuse if the answer isn't there."""
    context = "\n\n".join(c.page_content for c in chunks) if chunks else "(no documents found)"

    history_text = ""
    if history:
        history_text = "\n\nChat history:\n" + "\n".join(
            f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history
        )

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the documents below.
If the documents don't contain the answer, say "I don't have enough information to answer that."
Do not make things up.

Documents:
{context}
{history_text}

Question: {query}

Answer:"""
    return llm.invoke(prompt).content
