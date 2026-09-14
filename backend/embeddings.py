import hashlib
import numpy as np
from typing import List
from backend.config import GEMINI_API_KEY, EMBEDDING_MODEL

_GEMINI_AVAILABLE = False

try:
    from google import genai
    from google.genai import types as genai_types
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        _client = genai.Client(api_key=GEMINI_API_KEY)
        _GEMINI_AVAILABLE = True
    else:
        _client = None
except Exception:
    _client = None

def _mock_embedding(text: str, dim: int = 3072) -> np.ndarray:
    """Deterministic mock embedding for offline/test use."""
    seed = int(hashlib.sha256(text.encode('utf-8')).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec

def get_embedding(text: str) -> np.ndarray:
    """Generate normalized float32 embedding. Falls back to mock if no API key."""
    if not text:
        return np.zeros(3072, dtype=np.float32)
    if _GEMINI_AVAILABLE and _client:
        try:
            response = _client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text,
            )
            vec = np.array(response.embeddings[0].values, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception:
            pass
    return _mock_embedding(text)

from concurrent.futures import ThreadPoolExecutor

def get_query_embedding(query: str) -> np.ndarray:
    """Generate query embedding for semantic search."""
    return get_embedding(query)

def get_batch_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Generate embeddings for a list of texts in parallel for speed."""
    if not texts:
        return []
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(get_embedding, texts))
    return results
