"""PDF ingestion for the industrial cybersecurity (OT/ICS) RAG pipeline.

Usage:
    # Single PDF
    python -m app.rag.ingestion docs/pdfs/IEC62443-Overview.pdf

    # Directory of PDFs (batch)
    python -m app.rag.ingestion --dir docs/pdfs/

    # Custom collection and chunk params
    # python -m app.rag.ingestion --dir docs/pdfs/ --collection iec62443 --chunk-size 1000

Follows the pattern from example_RAG/5_vector_stores.py:
    - Connects to ChromaDB HTTP server (not embedded)
    - Deletes + recreates collection for clean reload
    - Uses PyPDFDirectoryLoader for batch loading
    - Verifies collection after insert
"""

import argparse
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings, get_settings
from app.rag.embeddings import NanBuildingsEmbeddings
from app.rag.exceptions import IngestionError


def load_pdfs_from_directory(dir_path: Path) -> list[Document]:
    """Load all PDFs from a directory using PyPDFDirectoryLoader."""
    if not dir_path.is_dir():
        raise IngestionError(f"Directory not found: {dir_path}")

    loader = PyPDFDirectoryLoader(str(dir_path))
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from directory: {dir_path}")
    return documents


def load_single_pdf(pdf_path: Path) -> list[Document]:
    """Load a single PDF using PyPDFLoader."""
    if not pdf_path.is_file():
        raise IngestionError(f"PDF file not found: {pdf_path}")

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"Loaded {len(documents)} pages from: {pdf_path.name}")
    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into chunks using recursive character splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def ingest_pdfs(
    pdf_dir: Path | None = None,
    pdf_path: Path | None = None,
    collection_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    settings: Settings | None = None,
) -> int:
    """Ingest PDFs into ChromaDB and return the number of chunks created.

    Connects to the ChromaDB HTTP server, deletes the existing collection
    for a clean reload, splits documents into chunks, generates embeddings,
    and stores them.
    """
    settings = settings or get_settings()
    collection_name = collection_name or settings.CHROMA_COLLECTION
    chunk_size = chunk_size or settings.CHUNK_SIZE
    chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

    # Load documents
    if pdf_dir:
        documents = load_pdfs_from_directory(pdf_dir)
    elif pdf_path:
        documents = load_single_pdf(pdf_path)
    else:
        raise IngestionError("Either --dir or a PDF path is required")

    if not documents:
        print("No documents found; nothing to ingest.")
        return 0

    # Split
    print(
        f"Splitting {len(documents)} pages into chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})..."
    )
    chunks = split_documents(documents, chunk_size, chunk_overlap)
    print(f"  → {len(chunks)} chunks created")

    # Connect to ChromaDB server
    print(f"Connecting to ChromaDB at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}...")
    chroma_client = chromadb.HttpClient(
        host=settings.CHROMA_HOST,
        port=settings.CHROMA_PORT,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # Delete collection if exists (clean reload)
    for col in chroma_client.list_collections():
        if col.name == collection_name:
            chroma_client.delete_collection(collection_name)
            print(f"  Deleted existing collection '{collection_name}' (clean reload)")

    # Generate embeddings and store
    print(f"Generating embeddings and storing in '{collection_name}'...")
    embeddings = NanBuildingsEmbeddings(settings=settings)

    Chroma.from_documents(
        documents=chunks,
        collection_name=collection_name,
        embedding=embeddings,
        client=chroma_client,
    )

    # Verify
    server_cols = chroma_client.list_collections()
    for c in server_cols:
        if c.name == collection_name:
            print(f"\n✓ {c.count()} chunks persisted in ChromaDB (collection '{c.name}')")

    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest OT/ICS cybersecurity PDFs into ChromaDB (HTTP server mode)."
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="Path to a single PDF file",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        help="Path to a directory of PDFs (batch mode)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name (default from settings)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Maximum chunk size in characters (default from settings)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Chunk overlap in characters (default from settings)",
    )
    args = parser.parse_args(argv)

    if not args.dir and not args.pdf:
        print("Error: specify either a PDF file or --dir with a directory", file=sys.stderr)
        return 1

    if args.pdf and not args.pdf.is_file():
        print(f"Error: file not found: {args.pdf}", file=sys.stderr)
        return 1

    if args.dir and not args.dir.is_dir():
        print(f"Error: directory not found: {args.dir}", file=sys.stderr)
        return 1

    try:
        count = ingest_pdfs(
            pdf_dir=args.dir,
            pdf_path=args.pdf,
            collection_name=args.collection,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    except IngestionError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1

    print(f"\nDone. {count} chunks ingested successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
