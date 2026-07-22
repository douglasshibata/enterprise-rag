import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Setup automatic mocking of external clients/heavy models for API tests
@pytest.fixture(autouse=True)
def mock_external_deps():
    with patch('app.main.os_client') as mock_os, \
         patch('app.main.generate_query_embedding') as mock_embed, \
         patch('app.main.generate_embeddings') as mock_embeds, \
         patch('app.main.fetch_url_content') as mock_fetch:

        # Configure standard mock return values
        mock_os.client = MagicMock()
        mock_os.client.info.return_value = {"version": "2.8.0-mock"}
        mock_os.bulk_index.return_value = (5, [])  # 5 success, 0 failed

        mock_embed.return_value = [0.1] * 384
        mock_embeds.return_value = [[0.1] * 384]
        mock_fetch.return_value = "Fetched mock content from URL"

        yield mock_os, mock_embed, mock_embeds, mock_fetch

# Lazy import of app after patching setup is complete
from app.main import app

client = TestClient(app)

def test_health_check_healthy(mock_external_deps):
    mock_os, _, _, _ = mock_external_deps
    # Ensure OpenSearch is healthy
    mock_os.client.info.return_value = {"status": "green"}

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["opensearch"] is True

def test_health_check_unhealthy(mock_external_deps):
    mock_os, _, _, _ = mock_external_deps
    # Simulate OpenSearch connection failure
    mock_os.client.info.side_effect = Exception("Connection refused")

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["opensearch"] is False

def test_prometheus_metrics():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text

def test_embed_unauthorized():
    # Call without API Key
    response = client.post("/embed", json={"text": "Hello"})
    assert response.status_code == 403
    assert "Invalid API Key" in response.json()["detail"]

    # Call with wrong API Key
    response = client.post("/embed", json={"text": "Hello"}, headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 403

def test_embed_authorized(mock_external_deps):
    _, mock_embed, _, _ = mock_external_deps
    response = client.post(
        "/embed",
        json={"text": "Hello world query"},
        headers={"X-API-Key": "default-secret-key"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "embedding" in data
    assert len(data["embedding"]) == 384
    mock_embed.assert_called_once_with("Hello world query")

def test_ingest_unauthorized():
    response = client.post("/ingest", json={"text": "sample text", "source": "test"})
    assert response.status_code == 403

def test_ingest_authorized_text(mock_external_deps):
    mock_os, _, mock_embeds, _ = mock_external_deps
    response = client.post(
        "/ingest",
        json={
            "text": "Sentence 1. Sentence 2. Sentence 3. Sentence 4. Sentence 5.",
            "source": "manual-text",
            "chunk_size": 3,
            "overlap": 1
        },
        headers={"X-API-Key": "default-secret-key"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Ingestion completed"
    assert "doc_id" in data
    assert data["chunks_processed"] > 0
    assert data["indexed_successfully"] == 5

    # Assert OpenSearch bulk indexing was called
    assert mock_os.bulk_index.called

def test_ingest_authorized_url(mock_external_deps):
    mock_os, _, _, mock_fetch = mock_external_deps
    response = client.post(
        "/ingest",
        json={
            "url": "https://example.com/test-article",
            "source": "web-url",
            "chunk_size": 2,
            "overlap": 0
        },
        headers={"X-API-Key": "default-secret-key"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Ingestion completed"
    mock_fetch.assert_called_once_with("https://example.com/test-article")
    assert mock_os.bulk_index.called

def test_ingest_invalid_request():
    # Ingest without both text and url should fail
    response = client.post(
        "/ingest",
        json={"source": "invalid-req"},
        headers={"X-API-Key": "default-secret-key"}
    )
    assert response.status_code == 400
    assert "Either 'text' or 'url' must be provided" in response.json()["detail"]
