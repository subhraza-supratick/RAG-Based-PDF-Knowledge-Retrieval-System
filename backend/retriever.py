import re
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from backend.database import get_all_chunks
from backend.embeddings import get_query_embedding
from backend.config import DEFAULT_TOP_K

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D NumPy arrays, handling dimension alignment if needed."""
    if vec_a.shape != vec_b.shape:
        min_dim = min(len(vec_a), len(vec_b))
        vec_a = vec_a[:min_dim]
        vec_b = vec_b[:min_dim]
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def retrieve_semantic_chunks(
    question: str,
    document_id: Optional[int] = None,
    top_k: int = DEFAULT_TOP_K
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Phase 6: Semantic Retrieval.
    Computes query vector embedding and ranks chunks using NumPy matrix operations for high performance.
    Returns (top_k_chunks, retrieval_latency_ms).
    """
    start_time = time.perf_counter()
    
    # 1. Fetch chunks from SQLite
    chunks = get_all_chunks(doc_id=document_id)
    if not chunks:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return [], elapsed_ms

    # 2. Embed user question
    query_vector = get_query_embedding(question)

    # 3. Vectorized matrix cosine similarity calculation
    embeddings = [c["embedding"] for c in chunks]
    emb_matrix = np.array(embeddings, dtype=np.float32)
    q_vec = np.array(query_vector, dtype=np.float32)

    if q_vec.ndim != 1:
        q_vec = q_vec.flatten()

    min_dim = min(q_vec.shape[0], emb_matrix.shape[1])
    q_vec = q_vec[:min_dim]
    emb_matrix = emb_matrix[:, :min_dim]

    q_norm = np.linalg.norm(q_vec)
    emb_norms = np.linalg.norm(emb_matrix, axis=1)

    if q_norm == 0:
        scores = np.zeros(len(chunks), dtype=float)
    else:
        denom = emb_norms * q_norm
        denom[denom == 0] = 1.0
        dots = np.dot(emb_matrix, q_vec)
        scores = dots / denom
        scores[emb_norms == 0] = 0.0

    scored_chunks = []
    for idx, c in enumerate(chunks):
        scored_chunks.append({
            "chunk_id": c["id"],
            "document_id": c["document_id"],
            "page": c["page_number"],
            "chunk_index": c["chunk_index"],
            "filename": c["filename"],
            "text": c["text"],
            "score": round(float(scores[idx]), 4)
        })

    # 4. Rank top-k chunks
    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored_chunks[:top_k]
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return top_chunks, round(elapsed_ms, 2)

def retrieve_baseline_keyword_chunks(
    question: str,
    document_id: Optional[int] = None,
    top_k: int = DEFAULT_TOP_K
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Phase 17: Baseline Keyword Retriever.
    Scans text chunks and ranks them based on keyword/token overlap.
    Returns (top_k_chunks, retrieval_latency_ms).
    """
    start_time = time.perf_counter()
    chunks = get_all_chunks(doc_id=document_id)
    if not chunks:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return [], elapsed_ms

    # Tokenize question into terms (removing short punctuation)
    q_tokens = set(re.findall(r'\b\w{2,}\b', question.lower()))

    scored_chunks = []
    for c in chunks:
        text_lower = c["text"].lower()
        chunk_tokens = re.findall(r'\b\w{2,}\b', text_lower)
        
        # Calculate term overlap count
        overlap = sum(1 for t in chunk_tokens if t in q_tokens)
        score = float(overlap) / (len(q_tokens) if q_tokens else 1.0)
        
        scored_chunks.append({
            "chunk_id": c["id"],
            "document_id": c["document_id"],
            "page": c["page_number"],
            "chunk_index": c["chunk_index"],
            "filename": c["filename"],
            "text": c["text"],
            "score": round(score, 4)
        })

    scored_chunks.sort(key=lambda x: x["score"], reverse=True)
    top_chunks = scored_chunks[:top_k]
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return top_chunks, round(elapsed_ms, 2)
