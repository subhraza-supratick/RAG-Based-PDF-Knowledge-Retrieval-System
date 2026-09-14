import re
from typing import List, Dict, Any, Tuple
import pypdf

class PDFExtractionError(Exception):
    """Custom exception for PDF extraction issues."""
    pass

def clean_text(raw_text: str) -> str:
    """
    Phase 2: Text Cleaning.
    Normalizes whitespace and removes null bytes while strictly preserving:
    - Financial table values (e.g., $2.4B, 14.2%, -$12M, (50.4))
    - Percentages, currency signs ($ € £ ¥), negatives, dates, row labels
    - Paragraph breaks (double linebreaks)
    """
    if not raw_text:
        return ""
    
    # Remove null bytes and non-printable control characters (except standard newlines/tabs)
    text = raw_text.replace('\x00', '')
    text = re.sub(r'[\r\f\v]', '\n', text)
    
    # Normalize spaces on each line while preserving numbers, signs, table columns
    lines = []
    for line in text.split('\n'):
        # Collapse multiple horizontal spaces to single space
        line_clean = re.sub(r'[ \t]+', ' ', line).strip()
        lines.append(line_clean)
    
    # Rejoin lines, replacing 3+ consecutive newlines with 2 newlines (preserving paragraphs)
    cleaned = '\n'.join(lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned

def extract_pdf_pages(file_path: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Phase 1: PDF Extraction.
    Extracts text page-by-page from a PDF file using PyPDF.
    Returns: (list_of_page_dicts, page_count)
    Each page dict contains: page_number (1-indexed), text, cleaned_text, character_count.
    
    Raises PDFExtractionError if encrypted, malformed, empty, or scanned/image-only without OCR.
    """
    try:
        reader = pypdf.PdfReader(file_path)
    except Exception as e:
        raise PDFExtractionError(f"Failed to open PDF file: {str(e)}")

    if reader.is_encrypted:
        try:
            # Try empty password decrypt if possible
            decrypted = reader.decrypt("")
            if not decrypted:
                raise PDFExtractionError("PDF is password-protected and cannot be read without a password.")
        except Exception:
            raise PDFExtractionError("PDF is password-protected and cannot be read without a password.")

    page_count = len(reader.pages)
    if page_count == 0:
        raise PDFExtractionError("PDF document is empty (0 pages).")

    extracted_pages = []
    total_chars = 0

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        try:
            raw_text = page.extract_text() or ""
        except Exception:
            raw_text = ""

        cleaned = clean_text(raw_text)
        char_count = len(cleaned)
        total_chars += char_count

        extracted_pages.append({
            "page_number": page_num,
            "raw_text": raw_text,
            "text": cleaned,
            "character_count": char_count
        })

    # Check for scanned or image-only PDF (very low character count per page)
    avg_chars = total_chars / page_count if page_count > 0 else 0
    if total_chars == 0 or avg_chars < 15:
        raise PDFExtractionError(
            "Scanned or image-only PDF detected (no readable text found). OCR processing is required."
        )

    return extracted_pages, page_count
