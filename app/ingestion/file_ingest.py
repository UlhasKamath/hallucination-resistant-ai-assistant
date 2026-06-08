"""
File ingestion pipeline.
Supports PDF, TXT, and DOCX uploads from Streamlit.
"""
import hashlib
import io

from app.ingestion.chunking import chunk_texts
from app.retrieval.dense import add_to_db, source_exists_in_chroma
from app.retrieval.sparse import add_to_sparse
from app.logging.logger import logger


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        return "\n".join(
            page.extract_text() or "" for page in reader.pages
        ).strip()
    except ImportError:
        raise ImportError("pypdf is required for PDF ingestion. Run: pip install pypdf")


def _extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        import docx
        from docx import Document
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs).strip()
    except ImportError:
        raise ImportError("python-docx is required for DOCX ingestion. Run: pip install python-docx")


def _extract_text(filename: str, file_bytes: bytes) -> str:
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        return _extract_text_from_pdf(file_bytes)
    elif ext in ("txt", "md"):
        return file_bytes.decode("utf-8", errors="ignore").strip()
    elif ext == "docx":
        return _extract_text_from_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: PDF, TXT, MD, DOCX")


def ingest_file(filename: str, file_bytes: bytes) -> dict:
    """
    Ingest an uploaded file into ChromaDB and the sparse index.
    Returns a summary dict with chunk count, doc_id, and status.
    """
    logger.info(f"FILE INGESTION STARTED -> {filename}")

    text = _extract_text(filename, file_bytes)

    if not text:
        return {"status": "error", "message": "Could not extract text from file."}

    doc_id       = _hash(text)
    source_label = f"file://{filename}"

    # Deduplicate using Chroma only
    if source_exists_in_chroma(source_label):
        logger.info(f"FILE ALREADY INDEXED -> {filename}")

        return {
            "status":  "duplicate",
            "message": f"{filename} has already been indexed.",
            "doc_id":  doc_id
    }


    # Chunk
    chunks_text = chunk_texts([text])
    chunks = [
        {
            "text":     c,
            "source":   source_label,
            "chunk_id": f"{doc_id}_{i}"
        }
        for i, c in enumerate(chunks_text)
    ]

    # Index into ChromaDB and sparse store
    add_to_db(chunks)
    add_to_sparse(chunks)

    logger.info(f"FILE INDEXED -> {filename} | {len(chunks)} chunks")

    return {
        "status":      "success",
        "filename":    filename,
        "doc_id":      doc_id,
        "chunk_count": len(chunks),
        "text":        text,
        "message":     f"Successfully indexed {len(chunks)} chunks from {filename}."
    }
