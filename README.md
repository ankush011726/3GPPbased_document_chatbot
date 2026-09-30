# RAG Chat

Ask questions about your PDFs and DOCX documents. Hugging Face embeds the documents, Gemini answers.

## How it works

```
docs/*.pdf, docs/*.docx
   │
   ▼
ingest.py          Parse documents (unstructured for images/tables, pypdf/docx2txt as fallback)
   │               → chunk → Hugging Face embeddings → Chroma (saved in db/)
   ▼
app.py             Streamlit chat
   │
   ├─ rewrite_query      follow-up questions made standalone using chat history
   ├─ Retriever          multi-query → hybrid (vector + BM25) → RRF → LLM rerank
   └─ answer_question    Gemini answers from retrieved chunks, or says it doesn't know
```

## Setup

1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

2. Add your API key to `.env`:
   ```
   GOOGLE_API_KEY=...   # https://aistudio.google.com/apikey
   ```

3. Put PDFs or DOCX files in `docs/`.

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
- Retrieved chunks are shown under each answer (click "Retrieved chunks") so you
  can see why the model answered the way it did.
- Settings like chunk size, model names and retrieval weights live in `config.py`.
