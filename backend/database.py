import sqlite3
import datetime
import hashlib
import json
import numpy as np
from typing import List, Dict, Any, Optional
from backend.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initialize SQLite database tables for documents, vector chunks, and query cache."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                page_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                page_number INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_hash TEXT NOT NULL,
                question TEXT NOT NULL,
                document_id INTEGER,
                top_k INTEGER NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                is_fallback INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_query_cache_hash ON query_cache(question_hash);
        """)
        conn.commit()

def insert_document(filename: str, page_count: int) -> int:
    """Insert document metadata and return document_id."""
    created_at = datetime.datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documents (filename, page_count, created_at) VALUES (?, ?, ?)",
            (filename, page_count, created_at)
        )
        conn.commit()
        return cursor.lastrowid

def insert_chunks(chunks_data: List[Dict[str, Any]]) -> None:
    """
    Insert a list of chunk dictionaries.
    Each dict must contain: document_id, page_number, chunk_index, text, embedding (np.ndarray or bytes)
    """
    records = []
    for c in chunks_data:
        emb = c["embedding"]
        if isinstance(emb, np.ndarray):
            emb_blob = emb.astype(np.float32).tobytes()
        elif isinstance(emb, (list, tuple)):
            emb_blob = np.array(emb, dtype=np.float32).tobytes()
        else:
            emb_blob = emb
        records.append((
            c["document_id"],
            c["page_number"],
            c["chunk_index"],
            c["text"],
            emb_blob
        ))
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO chunks (document_id, page_number, chunk_index, text, embedding) VALUES (?, ?, ?, ?, ?)",
            records
        )
        conn.commit()

def get_all_documents() -> List[Dict[str, Any]]:
    """Return all indexed documents."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, page_count, created_at FROM documents ORDER BY id DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_document_by_id(doc_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve document record by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, filename, page_count, created_at FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def delete_document(doc_id: int) -> bool:
    """Delete document and all associated chunks."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return cursor.rowcount > 0

def get_all_chunks(doc_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Retrieve chunk records (including document filename).
    Optionally filter by document_id.
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        if doc_id is not None:
            query = """
                SELECT c.id, c.document_id, c.page_number, c.chunk_index, c.text, c.embedding, d.filename
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE c.document_id = ?
                ORDER BY c.page_number, c.chunk_index
            """
            cursor.execute(query, (doc_id,))
        else:
            query = """
                SELECT c.id, c.document_id, c.page_number, c.chunk_index, c.text, c.embedding, d.filename
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                ORDER BY c.document_id, c.page_number, c.chunk_index
            """
            cursor.execute(query)
        
        rows = cursor.fetchall()
        result = []
        for r in rows:
            row_dict = dict(r)
            # deserialize embedding blob back to numpy float32 array
            row_dict["embedding"] = np.frombuffer(row_dict["embedding"], dtype=np.float32)
            result.append(row_dict)
        return result

def _hash_query(question: str, document_id: Optional[int], top_k: int) -> str:
    raw_str = f"{question.strip().lower()}:{document_id}:{top_k}"
    return hashlib.sha256(raw_str.encode('utf-8')).hexdigest()

def get_cached_answer(question: str, document_id: Optional[int], top_k: int) -> Optional[Dict[str, Any]]:
    """Retrieve cached query response if available."""
    q_hash = _hash_query(question, document_id, top_k)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT answer, sources_json, is_fallback FROM query_cache
            WHERE question_hash = ? ORDER BY id DESC LIMIT 1
        """, (q_hash,))
        row = cursor.fetchone()
        if row:
            sources = json.loads(row["sources_json"]) if row["sources_json"] else []
            return {
                "answer": row["answer"],
                "sources": sources,
                "is_fallback": bool(row["is_fallback"]),
                "cached": True,
                "metrics": {"retrieval_ms": 0.0, "generation_ms": 0.0, "cached": True}
            }
        return None

def save_cached_answer(
    question: str,
    document_id: Optional[int],
    top_k: int,
    answer: str,
    sources: List[Dict[str, Any]],
    is_fallback: bool = False
) -> None:
    """Store query answer and sources in query_cache table."""
    q_hash = _hash_query(question, document_id, top_k)
    sources_json = json.dumps(sources)
    created_at = datetime.datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO query_cache (question_hash, question, document_id, top_k, answer, sources_json, is_fallback, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (q_hash, question, document_id, top_k, answer, sources_json, 1 if is_fallback else 0, created_at))
        conn.commit()

def clear_query_cache(document_id: Optional[int] = None) -> None:
    """Invalidate cache entries for a specific document or all documents."""
    with get_connection() as conn:
        cursor = conn.cursor()
        if document_id is not None:
            cursor.execute("DELETE FROM query_cache WHERE document_id = ?", (document_id,))
        else:
            cursor.execute("DELETE FROM query_cache")
        conn.commit()
