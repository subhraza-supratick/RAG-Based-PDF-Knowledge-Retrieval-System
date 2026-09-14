import os
import sys
import sqlite3
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backend.database as db_mod
from backend.database import (
    init_db, insert_document, insert_chunks,
    get_all_documents, get_document_by_id, delete_document, get_all_chunks
)


@pytest.fixture(autouse=True)
def fresh_db(tmp_path):
    """Redirect all DB operations to a fresh temp SQLite file per test."""
    db_path = str(tmp_path / "test_rag.db")
    original_get_conn = db_mod.get_connection

    def mock_conn():
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn

    db_mod.get_connection = mock_conn
    init_db()
    yield
    db_mod.get_connection = original_get_conn


class TestDatabase:
    def test_insert_and_retrieve_document(self):
        doc_id = insert_document("report.pdf", 42)
        assert doc_id is not None
        doc = get_document_by_id(doc_id)
        assert doc['filename'] == "report.pdf"
        assert doc['page_count'] == 42
        assert 'created_at' in doc

    def test_list_documents_empty(self):
        docs = get_all_documents()
        assert docs == []

    def test_list_documents_after_insert(self):
        insert_document("a.pdf", 10)
        insert_document("b.pdf", 20)
        docs = get_all_documents()
        assert len(docs) == 2

    def test_insert_chunks_with_numpy_embedding(self):
        doc_id = insert_document("test.pdf", 5)
        vec = np.random.randn(768).astype(np.float32)
        chunks = [{
            "document_id": doc_id,
            "page_number": 1,
            "chunk_index": 0,
            "text": "Financial results show 12% growth.",
            "embedding": vec
        }]
        insert_chunks(chunks)
        retrieved = get_all_chunks(doc_id=doc_id)
        assert len(retrieved) == 1
        assert retrieved[0]['text'] == "Financial results show 12% growth."
        assert retrieved[0]['page_number'] == 1

    def test_embedding_roundtrip_numpy(self):
        doc_id = insert_document("embed.pdf", 3)
        original_vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        chunks = [{
            "document_id": doc_id,
            "page_number": 1,
            "chunk_index": 0,
            "text": "Test chunk.",
            "embedding": original_vec
        }]
        insert_chunks(chunks)
        retrieved = get_all_chunks(doc_id=doc_id)
        np.testing.assert_allclose(retrieved[0]['embedding'], original_vec, rtol=1e-5)

    def test_get_all_chunks_no_filter(self):
        doc1 = insert_document("doc1.pdf", 2)
        doc2 = insert_document("doc2.pdf", 3)
        vec = np.zeros(4, dtype=np.float32)
        insert_chunks([{"document_id": doc1, "page_number": 1, "chunk_index": 0, "text": "A", "embedding": vec}])
        insert_chunks([{"document_id": doc2, "page_number": 1, "chunk_index": 0, "text": "B", "embedding": vec}])
        all_chunks = get_all_chunks()
        assert len(all_chunks) == 2

    def test_delete_document_removes_chunks(self):
        doc_id = insert_document("del.pdf", 2)
        vec = np.zeros(768, dtype=np.float32)
        insert_chunks([{
            "document_id": doc_id,
            "page_number": 1,
            "chunk_index": 0,
            "text": "some text",
            "embedding": vec
        }])
        delete_document(doc_id)
        assert get_document_by_id(doc_id) is None
        assert get_all_chunks(doc_id=doc_id) == []

    def test_get_nonexistent_document(self):
        assert get_document_by_id(9999) is None

    def test_multiple_chunks_same_document(self):
        doc_id = insert_document("multi.pdf", 10)
        vec = np.ones(4, dtype=np.float32)
        chunks = [
            {"document_id": doc_id, "page_number": i, "chunk_index": i,
             "text": f"Chunk {i}", "embedding": vec}
            for i in range(5)
        ]
        insert_chunks(chunks)
        retrieved = get_all_chunks(doc_id=doc_id)
        assert len(retrieved) == 5
