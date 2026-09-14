import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.chunker import chunk_document_pages


SAMPLE_PAGES = [
    {"page_number": 1, "text": "Revenue grew by 14.2% in FY2024. " * 40, "character_count": 1000},
    {"page_number": 2, "text": "Operating expenses increased due to R&D investments. " * 30, "character_count": 800},
    {"page_number": 3, "text": "", "character_count": 0},  # blank page
]


class TestChunker:
    def test_returns_list_of_chunks(self):
        chunks = chunk_document_pages(SAMPLE_PAGES, document_id=1)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_each_chunk_has_required_fields(self):
        chunks = chunk_document_pages(SAMPLE_PAGES, document_id=1)
        for c in chunks:
            assert 'document_id' in c
            assert 'page_number' in c
            assert 'chunk_index' in c
            assert 'text' in c

    def test_document_id_preserved(self):
        chunks = chunk_document_pages(SAMPLE_PAGES, document_id=42)
        for c in chunks:
            assert c['document_id'] == 42

    def test_page_numbers_preserved(self):
        chunks = chunk_document_pages(SAMPLE_PAGES, document_id=1)
        page_nums = {c['page_number'] for c in chunks}
        assert 1 in page_nums
        assert 2 in page_nums
        # Page 3 is blank, should not appear
        assert 3 not in page_nums

    def test_blank_pages_skipped(self):
        chunks = chunk_document_pages(SAMPLE_PAGES, document_id=1)
        for c in chunks:
            assert c['text'].strip() != ''

    def test_chunk_index_is_sequential(self):
        chunks = chunk_document_pages(SAMPLE_PAGES, document_id=1)
        indices = [c['chunk_index'] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_size_respects_limit(self):
        chunks = chunk_document_pages(SAMPLE_PAGES, document_id=1, chunk_size=500, chunk_overlap=50)
        for c in chunks:
            # Allow small overrun due to splitter behavior
            assert len(c['text']) <= 600

    def test_empty_pages_produce_no_chunks(self):
        empty_pages = [{"page_number": 1, "text": "", "character_count": 0}]
        chunks = chunk_document_pages(empty_pages, document_id=1)
        assert chunks == []
