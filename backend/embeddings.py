import hashlib
import time
import numpy as np
from typing import List
from concurrent.futures import ThreadPoolExecutor
from backend.config import get_gemini_api_key, EMBEDDING_MODEL

def _get_genai_client():
    api_key = get_gemini_api_key()
    if api_key and api_key != "your_gemini_api_key_here":
        try:
            from google import genai
            return genai.Client(api_key=api_key)
        except Exception:
            return None
    return None

def _mock_embedding(text: str, dim: int = 3072) -> np.ndarray:
    """Deterministic mock embedding for offline/test use."""
    seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def get_embedding(text: str, retries: int = 3) -> np.ndarray:
    """Generate normalized float32 embedding. Retries on API rate limits."""
    if not text:
        return np.zeros(3072, dtype=np.float32)
    client = _get_genai_client()
    if client:
        for attempt in range(retries):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=text,
                )
                if response and hasattr(response, 'embeddings') and response.embeddings:
                    vec = np.array(response.embeddings[0].values, dtype=np.float32)
                    norm = np.linalg.norm(vec)
                    return vec / norm if norm > 0 else vec
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    time.sleep(1.2 * (attempt + 1))
                else:
                    time.sleep(0.3)
    return _mock_embedding(text)

def get_query_embedding(query: str) -> np.ndarray:
    """Generate query embedding for semantic search."""
    return get_embedding(query)

def get_batch_embeddings(texts: List[str], batch_size: int = 40) -> List[np.ndarray]:
    """Generate embeddings for a list of texts using batch API calls for fast performance."""
    if not texts:
        return []
    
    client = _get_genai_client()
    if not client:
        return [_mock_embedding(t) for t in texts]

    results: List[Optional[np.ndarray]] = [None] * len(texts)

    for i in range(0, len(texts), batch_size):
        batch_slice = texts[i:i + batch_size]
        batch_indices = list(range(i, min(i + batch_size, len(texts))))
        
        success = False
        for attempt in range(3):
            try:
                response = client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=batch_slice
                )
                if response and hasattr(response, 'embeddings') and len(response.embeddings) == len(batch_slice):
                    for idx, emb_obj in zip(batch_indices, response.embeddings):
                        vec = np.array(emb_obj.values, dtype=np.float32)
                        norm = np.linalg.norm(vec)
                        results[idx] = vec / norm if norm > 0 else vec
                    success = True
                    break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                    time.sleep(1.5 * (attempt + 1))
                else:
                    time.sleep(0.3)

        if not success:
            with ThreadPoolExecutor(max_workers=5) as executor:
                sub_res = list(executor.map(get_embedding, batch_slice))
                for idx, vec in zip(batch_indices, sub_res):
                    results[idx] = vec

    return [r if r is not None else _mock_embedding(texts[idx]) for idx, r in enumerate(results)]
