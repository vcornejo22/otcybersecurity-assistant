"""
Standalone PDF ingestion script for the industrial cybersecurity (OT/ICS) ChromaDB vector store.

Usage:
    python scripts/ingest.py

Follows the pattern from example_RAG/5_vector_stores.py:
    - Connects to ChromaDB HTTP server
    - Loads all PDFs from a directory using PyPDFDirectoryLoader
    - Splits with RecursiveCharacterTextSplitter
    - Deletes + recreates collection for clean reload
    - Generates qwen3-embedding embeddings via Nan Builders API
    - Verifies collection after insert

Requirements:
    pip install chromadb langchain-chroma langchain-community \
        langchain-text-splitters langchain-openai pymupdf python-dotenv

"""

import os
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Load environment ────────────────────────────────────────────
load_dotenv(override=True, verbose=True)

# ── Configuration ────────────────────────────────────────────────
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL_EMBEDDING = os.getenv("LLM_MODEL_EMBEDDING", "")
CHROMA_HOST = os.getenv("CHROMA_HOST", "")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", ""))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", ""))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", ""))
PDF_DIR = os.getenv("PDF_DIR", "docs/pdfs")

# ── Embeddings (Nan Builders API via OpenAI-compatible interface) ─
llm_embe = OpenAIEmbeddings(
    model=LLM_MODEL_EMBEDDING,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

# ── Connect to ChromaDB server ───────────────────────────────────
print(f"Connecting to ChromaDB at {CHROMA_HOST}:{CHROMA_PORT}...")
chroma_client = chromadb.HttpClient(
    host=CHROMA_HOST,
    port=CHROMA_PORT,
    settings=ChromaSettings(anonymized_telemetry=False),
)

# ── Load PDFs ────────────────────────────────────────────────────
pdf_dir = Path(PDF_DIR)
if not pdf_dir.is_dir():
    print(f"Error: PDF directory not found: {pdf_dir}")
    print("Place your industrial cybersecurity PDFs in docs/pdfs/ or set PDF_DIR in .env")
    exit(1)

loader = PyPDFDirectoryLoader(str(pdf_dir))
documents = loader.load()
print(f"Loaded {len(documents)} pages from {pdf_dir}")

if not documents:
    print("No PDFs found in the directory. Aborting.")
    exit(1)

# ── Split into chunks ────────────────────────────────────────────
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
docs_split = text_splitter.split_documents(documents=documents)
print(f"Split into {len(docs_split)} chunks")

# ── Delete existing collection (clean reload) ────────────────────
for col in chroma_client.list_collections():
    if col.name == CHROMA_COLLECTION:
        chroma_client.delete_collection(CHROMA_COLLECTION)
        print(f"Deleted existing collection '{CHROMA_COLLECTION}' (clean reload)")

# ── Generate embeddings & store (batched to avoid 413) ────────────
print(f"Generating {LLM_MODEL_EMBEDDING} embeddings and storing in '{CHROMA_COLLECTION}'...")
vector_store = Chroma(
    collection_name=CHROMA_COLLECTION,
    embedding_function=llm_embe,
    client=chroma_client,
)

BATCH_SIZE = 20  # small batches: ~320 KB per request
total = len(docs_split)
for i in range(0, total, BATCH_SIZE):
    batch = docs_split[i : i + BATCH_SIZE]
    texts = [d.page_content for d in batch]
    metadatas = [d.metadata for d in batch]
    ids = [f"chunk_{j}" for j in range(i, i + len(batch))]
    vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
    print(f"  {min(i + BATCH_SIZE, total)}/{total} chunks stored")

# ── Verify ───────────────────────────────────────────────────────
server_cols = chroma_client.list_collections()
for c in server_cols:
    if c.name == CHROMA_COLLECTION:
        print(f"\n✓ {c.count()} chunks persisted in ChromaDB (collection '{c.name}')")

# ── Quick test query ─────────────────────────────────────────────
question = "What are the security levels defined in industrial cybersecurity standards?"
print("\n--- Test query ---")
print(f"Q: {question}")

results = vector_store.similarity_search(query=question, k=2)
print(f"\nTop {len(results)} results:\n")
for i, doc in enumerate(results, start=1):
    print(f"--- Result {i} ---")
    src = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", "?")
    print(f"Source: {src} (page {page})")
    print(f"Content: {doc.page_content[:300]}...\n")
