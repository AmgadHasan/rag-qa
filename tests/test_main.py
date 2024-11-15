from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from src import models
from src.main import app

from . import PDF_FILE_PATH, CHUNKED_PDF_FILE_PATH, DOCUMENT_ID

client = TestClient(app)


@pytest.fixture
def mock_qdrant_client():
    return MagicMock(spec=QdrantClient)


@pytest.fixture
def mock_process_uploaded_file():
    with patch("src.main.process_uploaded_file") as mock:
        yield mock


@pytest.fixture
def mock_validate_pdf_file():
    with patch("src.main.validate_pdf_file") as mock:
        yield mock


@pytest.fixture
def mock_ingest_document():
    with patch("src.main.ingest_document") as mock:
        yield mock


@pytest.fixture
def mock_retrieve_relevant_context():
    with patch("src.main.retrieve_relevant_context") as mock:
        yield mock


@pytest.fixture
def mock_summarize_topic():
    with patch("src.main.summarize_topic") as mock:
        yield mock


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_ingest_pdf_success(
    mock_qdrant_client,
    mock_process_uploaded_file,
    mock_validate_pdf_file,
    mock_ingest_document,
):
    mock_process_uploaded_file.return_value = "processed_pdf"
    mock_ingest_document.return_value = models.DocumentMetadata(
        id="123", file_name="test.pdf"
    )

    with open(PDF_FILE_PATH, "rb") as f:
        response = client.post("/ingest", files={"file": f})

    assert response.status_code == 200
    assert response.json() == {"id": "123", "file_name": "test.pdf"}


def test_ingest_pdf_failure(
    mock_qdrant_client,
    mock_process_uploaded_file,
    mock_validate_pdf_file,
    mock_ingest_document,
):
    mock_process_uploaded_file.return_value = "processed_pdf"
    mock_ingest_document.side_effect = Exception("Failed to ingest document")

    with open("test.pdf", "wb") as f:
        f.write(b"fake pdf content")

    with open("test.pdf", "rb") as f:
        response = client.post("/ingest", files={"file": f})

    assert response.status_code == 500
    assert "Error processing PDF" in response.json()["detail"]


def test_generate_summary_success(
    mock_qdrant_client, mock_retrieve_relevant_context, mock_summarize_topic
):
    mock_retrieve_relevant_context.return_value = ["chunk1", "chunk2"]
    mock_summarize_topic.return_value = "Summary of the topic"

    response = client.post(
        "/generate/summary",
        json={
            "topic": "test topic",
            "document_id": "e2b9c8c1-7892-4dc9-b273-1f4c19802b1b",
        },
    )
    assert response.status_code == 200
    assert response.json() == {"topic": "test topic", "summary": "Summary of the topic"}


def test_generate_summary_failure(
    mock_qdrant_client, mock_retrieve_relevant_context, mock_summarize_topic
):
    mock_retrieve_relevant_context.side_effect = Exception("Failed to retrieve context")

    response = client.post(
        "/generate/summary",
        json={
            "topic": "test topic",
            "document_id": "e2b9c8c1-7892-4dc9-b273-1f4c19802b1b",
        },
    )

    assert response.status_code == 500
    assert "Error generating summary:" in response.json()["detail"]
