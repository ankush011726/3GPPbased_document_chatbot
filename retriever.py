"""
Retrieval pipeline: multi-query -> hybrid (vector + BM25) -> RRF fusion -> LLM rerank.
"""

import re
from collections import defaultdict

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

import config


def reciprocal_rank_fusion(result_lists, k=60):
    """Fuse several ranked lists into one. A chunk ranked high in many lists wins."""
    scores = defaultdict(float)
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results, start=1):
            scores[doc.page_content] += 1 / (k + rank)
            docs[doc.page_content] = doc
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[content] for content, _ in ranked]


class Retriever:
    def __init__(self, db):
        self.db = db
        self.llm = ChatGoogleGenerativeAI(model=config.CHAT_MODEL, temperature=0)

        # BM25 works on raw text, so pull the chunks back out of Chroma
        raw = db.get()
        chunks = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]
        self.bm25 = BM25Retriever.from_documents(chunks)
        self.bm25.k = 15

    def retrieve(self, query: str) -> list[Document]:
        """Full pipeline for one question. Returns the best chunks."""
        queries = self._expand_query(query)

        # Hybrid search: vector (semantic) + BM25 (exact keywords), weighted
        vector_retriever = self.db.as_retriever(search_kwargs={"k": 15})
        hybrid = EnsembleRetriever(
            retrievers=[vector_retriever, self.bm25],
            weights=[config.HYBRID_VECTOR_WEIGHT, config.HYBRID_BM25_WEIGHT],
        )

        # Search with every query variation, then fuse the rankings
        all_results = [hybrid.invoke(q) for q in queries]
        fused = reciprocal_rank_fusion(all_results)

        # LLM reranker picks the most relevant ones
        top = fused[: config.RERANK_TOP_N]
        return self._rerank(query, top)[: config.RETRIEVAL_K]

    def _expand_query(self, query: str) -> list[str]:
        """Ask the LLM for alternative phrasings of the question."""
        prompt = (
            f"Rewrite the following question in {config.MULTI_QUERY_COUNT} different ways "
            "that would help find the same information in documents. "
            "Return only the rewrites, one per line, with no numbers or bullets.\n\n"
            f"Question: {query}"
        )
        response = self.llm.invoke(prompt)
        variations = [line.strip(" -•\t") for line in response.content.split("\n") if line.strip()]
        return [query] + variations[: config.MULTI_QUERY_COUNT]

    def _rerank(self, query: str, chunks: list[Document]) -> list[Document]:
        """Ask the LLM to keep only relevant chunks, most relevant first."""
        if not chunks:
            return chunks

        numbered = "\n\n".join(f"[{i}] {c.page_content}" for i, c in enumerate(chunks))
        prompt = (
            "You are a search result ranker. Below is a question and numbered document chunks.\n"
            "Return the numbers of the chunks that help answer the question, most relevant first, "
            'as a comma-separated list (e.g. "2, 0, 3"). If none are relevant, return "none".\n\n'
            f"Question: {query}\n\nChunks:\n{numbered}\n\nRanked chunk numbers:"
        )
        response = self.llm.invoke(prompt)

        numbers = [int(n) for n in re.findall(r"\d+", response.content)]
        ranked = []
        for n in numbers:
            if 0 <= n < len(chunks) and chunks[n] not in ranked:
                ranked.append(chunks[n])
        return ranked
