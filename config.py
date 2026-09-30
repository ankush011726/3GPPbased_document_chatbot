"""Central settings for the RAG pipeline. Change things here, not in the code."""

# --- Paths ---
DOCS_DIR = "docs"
CHROMA_DIR = "db/chroma_db"

# --- Chunking ---
CHUNK_SIZE = 800        # characters per chunk
CHUNK_OVERLAP = 120     # overlap between chunks so context isn't cut off

# --- Embedding (Hugging Face) ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Generation (Gemini) ---
CHAT_MODEL = "gemini-2.5-flash"

# --- Retrieval ---
RETRIEVAL_K = 5                 # chunks passed to the answer generator
HYBRID_VECTOR_WEIGHT = 0.7      # vector vs BM25 weight in hybrid search
HYBRID_BM25_WEIGHT = 0.3
MULTI_QUERY_COUNT = 3           # query variations the LLM generates
RERANK_TOP_N = 10               # how many chunks the reranker keeps
