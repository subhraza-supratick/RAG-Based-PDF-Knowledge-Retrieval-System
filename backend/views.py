import os
import time
from pathlib import Path
from django.http import JsonResponse, HttpResponse, HttpRequest
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from backend.config import UPLOAD_DIR
from backend.database import (
    init_db, insert_document, insert_chunks,
    get_all_documents, get_document_by_id, delete_document, get_all_chunks
)
from backend.pdf_processor import extract_pdf_pages, PDFExtractionError
from backend.chunker import chunk_document_pages
from backend.embeddings import get_batch_embeddings
from backend.rag import generate_grounded_answer
from backend.schemas import validate_ask_request

# Ensure DB tables are initialized
init_db()

def serve_spa(request: HttpRequest):
    """Serve the Vanilla JS SPA single-page web app."""
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        with open(frontend_path, "r", encoding="utf-8") as f:
            return HttpResponse(f.read(), content_type="text/html")
    return HttpResponse("Frontend SPA index.html not found.", status=404)

@api_view(['GET'])
def health_check(request):
    """GET /api/health/ - Health check endpoint."""
    return Response({"status": "ok", "app": "Docsense Local AI (Django)"})

@api_view(['POST'])
def upload_document(request):
    """
    POST /api/documents/upload/
    Uploads a PDF document and creates metadata record.
    """
    if 'file' not in request.FILES:
        return Response({"error": "No file uploaded under key 'file'."}, status=status.HTTP_400_BAD_REQUEST)

    uploaded_file = request.FILES['file']
    filename = uploaded_file.name

    if not filename.lower().endswith('.pdf'):
        return Response({"error": "Only PDF files are supported."}, status=status.HTTP_400_BAD_REQUEST)

    # Sanitize filename
    safe_filename = Path(filename).name
    save_path = UPLOAD_DIR / safe_filename

    # Save to disk
    with open(save_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)

    try:
        pages, page_count = extract_pdf_pages(str(save_path))
    except PDFExtractionError as e:
        if save_path.exists():
            save_path.unlink()
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        return Response({"error": f"Failed to parse PDF: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Insert document into SQLite
    doc_id = insert_document(safe_filename, page_count)

    return Response({
        "document_id": doc_id,
        "filename": safe_filename,
        "page_count": page_count,
        "message": "File uploaded and verified successfully. Ready for indexing."
    }, status=status.HTTP_201_CREATED)

@api_view(['POST'])
def index_document(request, document_id: int):
    """
    POST /api/documents/<document_id>/index/
    Extracts pages, chunks text, computes embeddings, and indexes in SQLite.
    """
    doc = get_document_by_id(document_id)
    if not doc:
        return Response({"error": f"Document ID {document_id} not found."}, status=status.HTTP_404_NOT_FOUND)

    filepath = UPLOAD_DIR / doc["filename"]
    if not filepath.exists():
        return Response({"error": f"PDF file '{doc['filename']}' missing on server disk."}, status=status.HTTP_404_NOT_FOUND)

    try:
        start_time = time.perf_counter()
        pages, page_count = extract_pdf_pages(str(filepath))
        chunks_data = chunk_document_pages(pages, document_id)

        if not chunks_data:
            return Response({"error": "No valid text chunks created from document."}, status=status.HTTP_400_BAD_REQUEST)

        # Batch embed chunk texts
        texts = [c["text"] for c in chunks_data]
        embeddings = get_batch_embeddings(texts)

        for idx, emb in enumerate(embeddings):
            chunks_data[idx]["embedding"] = emb

        # Store in SQLite
        insert_chunks(chunks_data)
        elapsed_sec = round(time.perf_counter() - start_time, 2)

        return Response({
            "document_id": document_id,
            "filename": doc["filename"],
            "page_count": page_count,
            "total_chunks": len(chunks_data),
            "indexing_time_sec": elapsed_sec,
            "message": f"Successfully indexed {len(chunks_data)} chunks across {page_count} pages."
        }, status=status.HTTP_200_OK)

    except PDFExtractionError as e:
        return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({"error": f"Indexing failed: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['GET'])
def list_documents(request):
    """GET /api/documents/ - List all indexed documents."""
    docs = get_all_documents()
    return Response({"documents": docs}, status=status.HTTP_200_OK)

@api_view(['DELETE'])
def remove_document(request, document_id: int):
    """DELETE /api/documents/<document_id>/ - Delete document and associated chunks."""
    doc = get_document_by_id(document_id)
    if not doc:
        return Response({"error": f"Document ID {document_id} not found."}, status=status.HTTP_404_NOT_FOUND)

    # Remove from disk if present
    filepath = UPLOAD_DIR / doc["filename"]
    if filepath.exists():
        try:
            filepath.unlink()
        except Exception:
            pass

    deleted = delete_document(document_id)
    if deleted:
        return Response({"message": f"Document ID {document_id} deleted successfully."}, status=status.HTTP_200_OK)
    return Response({"error": "Failed to delete document."}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

@api_view(['POST'])
def ask_question(request):
    """
    POST /api/ask/
    Body: { "question": "...", "document_id": optional_int, "top_k": 5 }
    Returns grounded Gemini answer, citations with exact page numbers, and metrics.
    """
    is_valid, err_msg, question, doc_id, top_k = validate_ask_request(request.data)
    if not is_valid:
        return Response({"error": err_msg}, status=status.HTTP_400_BAD_REQUEST)

    result = generate_grounded_answer(question, document_id=doc_id, top_k=top_k)
    return Response(result, status=status.HTTP_200_OK)
