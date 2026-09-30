"""
Ingestion pipeline: docs/ -> chunks -> Gemini embeddings -> Chroma.

Run this whenever you add or change documents:
    python ingest.py

PDFs are parsed with unstructured (images + tables) when it is installed,
otherwise it falls back to pypdf (text-only PDFs).
DOCX files are parsed with unstructured when available, otherwise docx2txt.
"""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

load_dotenv()

try:
    from unstructured.chunking.title import chunk_by_title
    from unstructured.partition.docx import partition_docx
    from unstructured.partition.pdf import partition_pdf

    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False


def load_pdf(path: str, filename: str) -> list[Document]:
    """Load one PDF. Uses unstructured when available, else pypdf."""
    if HAS_UNSTRUCTURED:
        return _load_with_unstructured(path, filename)
    return _load_with_pypdf(path, filename)


def load_docx(path: str, filename: str) -> list[Document]:
    """Load one DOCX. Uses unstructured when available, else docx2txt."""
    if HAS_UNSTRUCTURED:
        return _load_docx_with_unstructured(path, filename)
    return _load_docx_with_docx2txt(path, filename)


def _load_with_unstructured(path: str, filename: str) -> list[Document]:
    """hi_res strategy keeps tables as structured text and pulls out images."""
    print(f"  Parsing {filename} with unstructured (hi_res)...")
    elements = partition_pdf(
        filename=path,
        strategy="hi_res",
        infer_table_structure=True,
    )
    chunks = chunk_by_title(
        elements,
        max_characters=config.CHUNK_SIZE,
        new_after_n_chars=int(config.CHUNK_SIZE * 0.8),
        combine_text_under_n_chars=200,
    )
    return [
        Document(page_content=str(chunk), metadata={"source": filename})
        for chunk in chunks
    ]


def _load_docx_with_unstructured(path: str, filename: str) -> list[Document]:
    """Parse docx with unstructured, chunk by title."""
    print(f"  Parsing {filename} with unstructured (docx)...")
    elements = partition_docx(filename=path)
    chunks = chunk_by_title(
        elements,
        max_characters=config.CHUNK_SIZE,
        new_after_n_chars=int(config.CHUNK_SIZE * 0.8),
        combine_text_under_n_chars=200,
    )
    return [
        Document(page_content=str(chunk), metadata={"source": filename})
        for chunk in chunks
    ]


def _load_with_pypdf(path: str, filename: str) -> list[Document]:
    """Plain text extraction, one Document per page."""
    print(f"  Loading {filename} with pypdf...")
    loader = PyPDFLoader(path)
    documents = []
    for page in loader.load():
        page.metadata["source"] = filename
        page.metadata["page"] = page.metadata.get("page", 0) + 1
        documents.append(page)
    return documents


def _load_docx_with_docx2txt(path: str, filename: str) -> list[Document]:
    """Plain text extraction with docx2txt, one Document per file."""
    print(f"  Loading {filename} with docx2txt...")
    loader = Docx2txtLoader(path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = filename
    return documents


def load_documents(docs_dir: str) -> list[Document]:
    """Load every PDF and DOCX in the docs folder."""
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"The folder '{docs_dir}' does not exist.")

    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".pdf")]
    docx_files = [f for f in os.listdir(docs_dir) if f.lower().endswith(".docx")]
    all_files = pdf_files + docx_files

    if not all_files:
        raise FileNotFoundError(
            f"No PDFs or DOCX files found in '{docs_dir}'. Add some and try again."
        )

    documents = []
    for filename in pdf_files:
        documents.extend(load_pdf(os.path.join(docs_dir, filename), filename))
    for filename in docx_files:
        documents.extend(load_docx(os.path.join(docs_dir, filename), filename))

    print(
        f"  Loaded {len(documents)} chunk(s) from {len(pdf_files)} PDF(s) "
        f"and {len(docx_files)} DOCX file(s)."
    )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split into overlapping chunks (only needed for pypdf pages)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Split into {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}).")
    return chunks


def create_vector_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and save them to disk."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=config.CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"  Saved vector store to {config.CHROMA_DIR}.")
    return vectorstore


def load_vector_store() -> Chroma:
    """Load the existing vector store without re-embedding anything."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return Chroma(
        persist_directory=config.CHROMA_DIR,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )


def main():
    print("=== Ingestion ===\n")

    if os.path.exists(config.CHROMA_DIR):
        db = load_vector_store()
        print(f"Vector store already exists ({db._collection.count()} chunks). Skipping ingestion.")
        print("Delete the db/ folder if you want to rebuild from scratch.\n")
        return

    documents = load_documents(config.DOCS_DIR)

    # unstructured already chunks by title; pypdf pages need splitting
    if HAS_UNSTRUCTURED:
        chunks = documents
    else:
        chunks = split_documents(documents)

    create_vector_store(chunks)

    print("\nDone. Your documents are ready for questions.")


if __name__ == "__main__":
    main()
