from typing import List, Dict, Any
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.config import CHUNK_SIZE, CHUNK_OVERLAP

def chunk_document_pages(
    pages: List[Dict[str, Any]],
    document_id: int,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict[str, Any]]:
    """
    Phase 3: Chunking.
    Splits page-level text into chunks while preserving metadata.
    
    Critical Rule: Every chunk retains its document_id, page_number, chunk_index, and text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    all_chunks: List[Dict[str, Any]] = []
    chunk_index = 0

    for p in pages:
        page_num = p["page_number"]
        page_text = p["text"]
        
        if not page_text or not page_text.strip():
            continue
            
        # Split text within the current page
        split_texts = splitter.split_text(page_text)
        
        for text_chunk in split_texts:
            if not text_chunk.strip():
                continue
            all_chunks.append({
                "document_id": document_id,
                "page_number": page_num,
                "chunk_index": chunk_index,
                "text": text_chunk.strip()
            })
            chunk_index += 1

    return all_chunks
