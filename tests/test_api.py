import os
import sys
import json
import django
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'document_rag.settings')
django.setup()

from django.test import Client
from backend.database import init_db

init_db()

@pytest.fixture
def client():
    return Client()


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        res = client.get('/api/health/')
        assert res.status_code == 200

    def test_health_returns_ok(self, client):
        res = client.get('/api/health/')
        data = json.loads(res.content)
        assert data.get('status') == 'ok'
        assert 'app' in data


class TestDocumentEndpoints:
    def test_list_documents_returns_200(self, client):
        res = client.get('/api/documents/')
        assert res.status_code == 200

    def test_list_documents_has_documents_key(self, client):
        res = client.get('/api/documents/')
        data = json.loads(res.content)
        assert 'documents' in data
        assert isinstance(data['documents'], list)

    def test_upload_no_file_returns_400(self, client):
        res = client.post('/api/documents/upload/')
        assert res.status_code == 400

    def test_upload_non_pdf_returns_400(self, client):
        from io import BytesIO
        fake_file = BytesIO(b"not a pdf")
        fake_file.name = "test.txt"
        res = client.post('/api/documents/upload/', {'file': fake_file})
        assert res.status_code == 400

    def test_delete_nonexistent_returns_404(self, client):
        res = client.delete('/api/documents/99999/')
        assert res.status_code == 404

    def test_index_nonexistent_returns_404(self, client):
        res = client.post('/api/documents/99999/index/')
        assert res.status_code == 404


class TestAskEndpoint:
    def test_ask_missing_question_returns_400(self, client):
        res = client.post(
            '/api/ask/',
            data=json.dumps({}),
            content_type='application/json'
        )
        assert res.status_code == 400

    def test_ask_empty_question_returns_400(self, client):
        res = client.post(
            '/api/ask/',
            data=json.dumps({"question": "   "}),
            content_type='application/json'
        )
        assert res.status_code == 400

    def test_ask_returns_answer_key(self, client):
        res = client.post(
            '/api/ask/',
            data=json.dumps({"question": "What is the revenue?"}),
            content_type='application/json'
        )
        assert res.status_code == 200
        data = json.loads(res.content)
        assert 'answer' in data
        assert 'sources' in data
        assert 'metrics' in data

    def test_ask_metrics_have_latency_fields(self, client):
        res = client.post(
            '/api/ask/',
            data=json.dumps({"question": "Summarize the document."}),
            content_type='application/json'
        )
        assert res.status_code == 200
        data = json.loads(res.content)
        assert 'retrieval_ms' in data['metrics']
        assert 'generation_ms' in data['metrics']
