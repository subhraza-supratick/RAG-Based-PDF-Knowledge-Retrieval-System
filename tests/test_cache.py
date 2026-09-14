import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import (
    init_db, insert_document, insert_chunks, get_cached_answer, save_cached_answer, clear_query_cache
)
from backend.rag import generate_grounded_answer

init_db()

@pytest.fixture(autouse=True)
def clean_cache():
    clear_query_cache()
    yield
    clear_query_cache()

class TestQueryMemorySystem:
    def test_save_and_get_cached_answer(self):
        question = "What is the revenue?"
        doc_id = 100
        top_k = 5
        answer = "Revenue was $10M."
        sources = [{"chunk_id": 1, "page": 1, "filename": "test.pdf", "text": "Revenue $10M", "score": 0.9}]

        # Before saving -> cache miss
        assert get_cached_answer(question, doc_id, top_k) is None

        # Save to cache
        save_cached_answer(question, doc_id, top_k, answer, sources)

        # After saving -> cache hit
        cached = get_cached_answer(question, doc_id, top_k)
        assert cached is not None
        assert cached["answer"] == answer
        assert cached["cached"] is True
        assert cached["metrics"]["retrieval_ms"] == 0.0
        assert len(cached["sources"]) == 1

    def test_clear_query_cache(self):
        save_cached_answer("Q1", 101, 5, "Ans1", [])
        save_cached_answer("Q2", 102, 5, "Ans2", [])

        assert get_cached_answer("Q1", 101, 5) is not None
        assert get_cached_answer("Q2", 102, 5) is not None

        # Clear cache for doc 101
        clear_query_cache(101)
        assert get_cached_answer("Q1", 101, 5) is None
        assert get_cached_answer("Q2", 102, 5) is not None

        # Clear all cache
        clear_query_cache()
        assert get_cached_answer("Q2", 102, 5) is None

class TestAPIFallbackSystem:
    def test_fallback_when_llm_api_raises_exception(self):
        from backend.embeddings import get_query_embedding
        question = "What is the revenue?"
        q_emb = get_query_embedding(question)

        # Insert test document & chunk
        doc_id = insert_document("sample_report.pdf", 1)
        insert_chunks([{
            "document_id": doc_id,
            "page_number": 1,
            "chunk_index": 0,
            "text": "The company reported total quarterly revenue of $50 million.",
            "embedding": q_emb
        }])

        with patch('backend.rag._client') as mock_client, \
             patch('backend.rag._GEMINI_LLM_AVAILABLE', True):
            mock_client.models.generate_content.side_effect = Exception("429 RESOURCE_EXHAUSTED: Rate limit exceeded")

            result = generate_grounded_answer(question, document_id=doc_id)

            assert "answer" in result
            assert "Offline Context Summary" in result["answer"]
            assert result["is_fallback"] is True
            assert len(result["sources"]) > 0

    def test_memory_cache_serves_cached_result_without_api_call(self):
        from backend.embeddings import get_query_embedding
        question = "How much did net income grow?"
        q_emb = get_query_embedding(question)

        doc_id = insert_document("cached_doc.pdf", 1)
        insert_chunks([{
            "document_id": doc_id,
            "page_number": 1,
            "chunk_index": 0,
            "text": "Net income grew by 25 percent year-over-year.",
            "embedding": q_emb
        }])

        # First call populates cache
        res1 = generate_grounded_answer(question, document_id=doc_id)
        assert "answer" in res1

        # Second identical call must return cached result
        res2 = generate_grounded_answer(question, document_id=doc_id)
        assert res2.get("cached") is True
        assert res2["metrics"]["retrieval_ms"] == 0.0
        assert res2["metrics"]["generation_ms"] == 0.0
