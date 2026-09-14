from typing import List, Optional, Dict, Any, Tuple

def validate_ask_request(data: Dict[str, Any]) -> Tuple[bool, str, str, Optional[int], int]:
    """Helper validator for /api/ask payload."""
    question = data.get("question", "").strip()
    if not question:
        return False, "Question is required.", "", None, 5
    
    doc_id = data.get("document_id")
    if doc_id is not None:
        try:
            doc_id = int(doc_id)
        except ValueError:
            return False, "Invalid document_id integer.", "", None, 5
            
    top_k = data.get("top_k", 5)
    try:
        top_k = int(top_k)
    except ValueError:
        top_k = 5
        
    return True, "", question, doc_id, top_k
