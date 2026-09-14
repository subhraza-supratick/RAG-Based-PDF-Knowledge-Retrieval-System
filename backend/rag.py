import time
from typing import List, Dict, Any, Optional
from backend.config import GEMINI_API_KEY, LLM_MODEL
from backend.retriever import retrieve_semantic_chunks

_GEMINI_LLM_AVAILABLE = False
_client = None

try:
    from google import genai
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        _client = genai.Client(api_key=GEMINI_API_KEY)
        _GEMINI_LLM_AVAILABLE = True
except Exception:
    pass

SYSTEM_PROMPT_TEMPLATE = """You are an expert document QA assistant.
Answer using ONLY the supplied context. Do not invent facts.
If evidence is insufficient, state clearly that the document does not contain enough information.

FORMATTING REQUIREMENTS FOR MAXIMUM READABILITY:
1. Provide a direct 1-sentence summary at the very beginning.
2. Present key points using clear **bullet points** or numbered lists. Do NOT write one massive continuous paragraph.
3. Use **bold text** to highlight key terms, metrics, numbers, or concepts.
4. Mention supporting page numbers (e.g. [Page X]) alongside key claims.

CONTEXT:
{context_text}

QUESTION:
{question}
"""

def format_context_blocks(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into [Page X] context blocks."""
    if not chunks:
        return ""
    blocks = []
    for c in chunks:
        blocks.append(f"[Page {c['page']}] ({c['filename']}):\n{c['text']}")
    return "\n\n".join(blocks)

def generate_grounded_answer(
    question: str,
    document_id: Optional[int] = None,
    top_k: int = 5
) -> Dict[str, Any]:
    """Retrieve top-k chunks and generate a grounded Gemini answer with page citations."""
    chunks, retrieval_ms = retrieve_semantic_chunks(question, document_id=document_id, top_k=top_k)

    if not chunks or chunks[0]["score"] < 0.05:
        return {
            "answer": "I could not find sufficient evidence in the document to answer your question.",
            "sources": [],
            "metrics": {"retrieval_ms": retrieval_ms, "generation_ms": 0.0}
        }

    context_str = format_context_blocks(chunks)
    prompt = SYSTEM_PROMPT_TEMPLATE.format(context_text=context_str, question=question)

    gen_start = time.perf_counter()
    answer_text = ""

    if _GEMINI_LLM_AVAILABLE and _client:
        try:
            response = _client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt
            )
            answer_text = response.text.strip() if response.text else ""
        except Exception as e:
            answer_text = f"Gemini API error: {str(e)}"

    if not answer_text:
        top_text = chunks[0]["text"]
        sentences = [s.strip() for s in top_text.split('.') if len(s.strip()) > 10]
        answer_text = f"Based on page {chunks[0]['page']}: {sentences[0]}." if sentences else top_text[:200]

    generation_ms = round((time.perf_counter() - gen_start) * 1000.0, 2)

    sources = [
        {
            "chunk_id": c["chunk_id"],
            "page": c["page"],
            "filename": c["filename"],
            "text": c["text"],
            "score": c["score"]
        }
        for c in chunks
    ]

    return {
        "answer": answer_text,
        "sources": sources,
        "metrics": {"retrieval_ms": retrieval_ms, "generation_ms": generation_ms}
    }
