import sqlite3
import datetime
import numpy as np
from typing import List, Dict, Any, Optional
from backend.config import DB_PATH

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    """Initialize SQLite database tables for documents and vector chunks."""
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
