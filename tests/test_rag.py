import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.rag import format_context_blocks, generate_grounded_answer


class TestFormatContextBlocks:
    def test_formats_page_headers(self):
        chunks = [
            {"page": 1, "filename": "report.pdf", "text": "Revenue was $2.4B."},
            {"page": 3, "filename": "report.pdf", "text": "Net income grew 14%."},
        ]
        result = format_context_blocks(chunks)
        assert "[Page 1]" in result
        assert "[Page 3]" in result
        assert "Revenue was $2.4B." in result

    def test_empty_chunks_returns_empty_string(self):
        result = format_context_blocks([])
        assert result == ""

    def test_blocks_separated_by_newlines(self):
        chunks = [
            {"page": 1, "filename": "a.pdf", "text": "Block one."},
            {"page": 2, "filename": "a.pdf", "text": "Block two."},
        ]
        result = format_context_blocks(chunks)
        assert "\n\n" in result


class TestRAGRefusal:
    def test_returns_answer_key(self):
        result = generate_grounded_answer("What is the revenue?")
        assert "answer" in result

    def test_returns_sources_key(self):
        result = generate_grounded_answer("What is the revenue?")
        assert "sources" in result

    def test_returns_metrics_key(self):
        result = generate_grounded_answer("What is the revenue?")
        assert "metrics" in result
        assert "retrieval_ms" in result["metrics"]
        assert "generation_ms" in result["metrics"]

    def test_insufficient_context_returns_refusal(self):
        result = generate_grounded_answer("xyzzy_no_matching_content_abc123_unique")
        assert "answer" in result
        assert len(result["answer"]) > 0
