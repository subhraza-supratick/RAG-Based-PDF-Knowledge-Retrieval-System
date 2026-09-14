import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.pdf_processor import extract_pdf_pages, clean_text, PDFExtractionError


class TestCleanText:
    def test_removes_null_bytes(self):
        result = clean_text("Hello\x00World")
        assert '\x00' not in result

    def test_preserves_financial_values(self):
        text = "Revenue: $2.4B  Growth: 14.2%  Loss: -$12M  Item: (50.4)"
        result = clean_text(text)
        assert '$2.4B' in result
        assert '14.2%' in result
        assert '-$12M' in result
        assert '(50.4)' in result

    def test_collapses_multiple_spaces(self):
        result = clean_text("Hello    World")
        assert "Hello World" in result

    def test_normalizes_excessive_newlines(self):
        result = clean_text("Para1\n\n\n\n\nPara2")
        assert "\n\n\n" not in result

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    def test_strips_leading_trailing_whitespace(self):
        result = clean_text("   hello   ")
        assert result == "hello"


class TestExtractPDFPages:
    def test_raises_on_nonexistent_file(self):
        with pytest.raises(PDFExtractionError):
            extract_pdf_pages("/nonexistent/path/file.pdf")

    def test_raises_on_non_pdf_file(self):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False, mode='wb') as f:
            f.write(b"This is not a real PDF file content")
            tmp_path = f.name
        try:
            with pytest.raises(PDFExtractionError):
                extract_pdf_pages(tmp_path)
        finally:
            os.unlink(tmp_path)
