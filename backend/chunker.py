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
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_len: int = 60
) -> List[Dict[str, Any]]:
    """
    Phase 3: Chunking.
    Splits page-level text into chunks while preserving metadata.
    Merges tiny low-information header/footer fragments (< min_chunk_len)
    into adjacent contextual page chunks.
    
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
        merged_texts: List[str] = []
        buffer = ""

        for text_chunk in split_texts:
            clean_chunk = text_chunk.strip()
            if not clean_chunk:
                continue

            if buffer:
                buffer = buffer + "\n" + clean_chunk
                if len(buffer) >= min_chunk_len:
                    merged_texts.append(buffer)
                    buffer = ""
            else:
                if len(clean_chunk) < min_chunk_len:
                    buffer = clean_chunk
                else:
                    merged_texts.append(clean_chunk)

        if buffer:
            if merged_texts:
                merged_texts[-1] = merged_texts[-1] + "\n" + buffer
            elif len(buffer) >= 20:
                merged_texts.append(buffer)

        for text_chunk in merged_texts:
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
