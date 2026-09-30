# RAG Chat

Ask questions about your PDFs. Gemini embeds the documents, Groq answers.

## How it works

```
docs/*.pdf
   │
   ▼
ingest.py          Parse PDFs (unstructured for images/tables, pypdf as fallback)
   │               → chunk → Gemini embeddings → Chroma (saved in db/)
   ▼
app.py             Streamlit chat
   │
   ├─ rewrite_query      follow-up questions made standalone using chat history
   ├─ Retriever          multi-query → hybrid (vector + BM25) → RRF → LLM rerank
   └─ answer_question    Groq answers from retrieved chunks, or says it doesn't know
```

## Setup

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

2. Add your API keys to `.env`:
   ```
   GEMINI_API_KEY=...   # https://aistudio.google.com/apikey
   GROQ_API_KEY=...    # https://console.groq.com
   ```

3. Put PDFs in `docs/`.

4. Build the vector store:
   ```powershell
   python ingest.py
   ```

5. Chat:
   ```powershell
   streamlit run app.py
   ```

## Notes

- **unstructured** needs system tools (poppler, tesseract) for hi_res PDF parsing.
  If they are missing, ingest.py falls back to pypdf (text-only PDFs).
  On Windows: `conda install -c conda-forge poppler tesseract` or skip unstructured
  and use text-based PDFs.
- Delete `db/` and re-run `python ingest.py` to rebuild the index from scratch.
- Retrieved chunks are shown under each answer (click "📄 Retrieved chunks") so you
  can see why the model answered the way it did.
- Settings like chunk size, model names and retrieval weights live in `config.py`.
