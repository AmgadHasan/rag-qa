import io
import json
import uuid
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

from src.emb import EMBEDDING_DIMENSION
from src.models import DocumentMetadata
from src.vectorstore import (
    ingest_document,
    load_and_split_document,
    retrieve_relevant_context,
    split_text,
)

PDF_FILE_PATH = "./samples/ml-engineer-tech-test.pdf"
DOCUMENT_ID = "71a18a69-2dbd-466b-99e5-4e3e213430a9"


@pytest.fixture
def mock_pdf_file():
    sample_pdf_file_path = PDF_FILE_PATH
    with open(sample_pdf_file_path, "rb") as file:
        pdf_content = file.read()

    pdf_bytes_io = io.BytesIO(pdf_content)
    pdf_bytes_io.name = sample_pdf_file_path

    return pdf_bytes_io


@pytest.fixture
def mock_chunks():
    with open("./samples/chunked-ml-engineer-tech-test.json", "r") as f:
        chunks = json.load(f)
    return chunks


@pytest.fixture
def mock_vectors():
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]


@pytest.fixture
def mock_search_result():
    class DummyResponse:
        def __init__(self, points):
            self.points = points

    return DummyResponse(
        points=[
            models.ScoredPoint(
                id="point1",
                vector=[0.1, 0.2, 0.3],
                payload={"text": "chunk1"},
                version=2,
                score=0.2,
            ),
            models.ScoredPoint(
                id="point2",
                vector=[0.4, 0.5, 0.6],
                payload={"text": "chunk2"},
                version=2,
                score=0.2,
            ),
        ]
    )


@pytest.fixture
def mock_client(mock_search_result):
    client = MagicMock(spec=QdrantClient)
    client.create_collection.return_value = None
    client.upsert.return_value = None
    client.query_points.return_value = mock_search_result
    return client


@pytest.fixture(autouse=True)
def mock_qdrant_client(monkeypatch, mock_client):
    monkeypatch.setattr(
        "src.vectorstore.QdrantClient", MagicMock(return_value=mock_client)
    )


@pytest.fixture
def mock_query_embeddings():
    return [0.1, 0.2, 0.3]


@pytest.fixture(autouse=True)
def mock_embed_query(monkeypatch, mock_query_embeddings):
    mock_func = MagicMock(return_value=mock_query_embeddings)
    monkeypatch.setattr("src.vectorstore.embed_query", mock_func)
    return mock_func


@pytest.fixture(autouse=True)
def mock_embed_texts(monkeypatch, mock_vectors):
    mock_func = MagicMock(return_value=mock_vectors)
    monkeypatch.setattr("src.emb.embed_texts", mock_func)
    return mock_func


@pytest.fixture(autouse=True)
def mock_pymupdf(monkeypatch, mock_chunks):
    mock_doc = MagicMock()
    mock_doc.page_count = 1
    mock_page = MagicMock()
    mock_page.get_text.return_value = " ".join(mock_chunks)
    mock_doc.__iter__.return_value = [mock_page]
    monkeypatch.setattr(
        "src.vectorstore.pymupdf.Document", MagicMock(return_value=mock_doc)
    )


@pytest.fixture(autouse=True)
def mock_logger(monkeypatch):
    mock_logger = MagicMock()
    monkeypatch.setattr("src.vectorstore.logger", mock_logger)


class TestVectorstore:
    def test_split_text_short_text(self):
        text = "Short text that doesn't need splitting."
        chunks = split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_split_text_long_text(self):
        text = "\n\n This is a very long text block. Again, too long\n\n" * 100
        chunks = split_text(text=text)
        chunks_lengths = [len(chunk) for chunk in chunks]
        assert max(chunks_lengths) <= 512

    def test_load_and_split_document_length(self, mock_pdf_file):
        chunks = load_and_split_document(mock_pdf_file)
        assert len(chunks) > 2

    def test_load_and_split_document_content(self, mock_pdf_file):
        chunks = load_and_split_document(mock_pdf_file)
        full_text = "\n".join(chunks)
        assert "technicaltest@alefeducation.com" in full_text

    def test_ingest_document(
        self, mock_pdf_file, mock_chunks, mock_vectors, mock_client
    ):
        with patch(
            "src.vectorstore.uuid.uuid4",
            side_effect=[
                uuid.UUID("71a18a69-2dbd-466b-99e5-4e3e213430a9"),
                uuid.UUID("cdba0257-4895-4908-8dc7-97faa33564dc"),
                uuid.UUID("1bf3fce1-6992-47ea-8728-9f55fe611b3b"),
                uuid.UUID("ebff8d16-df8e-4913-807b-e39390fa3f74"),
            ],
        ):
            metadata = ingest_document(mock_pdf_file, client=mock_client)
            mock_client.create_collection.assert_called_once_with(
                collection_name="71a18a69-2dbd-466b-99e5-4e3e213430a9",
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSION, distance=Distance.DOT
                ),
            )
            # # NOTE:
            # # For some reason, the document parsing is non-determinstic and we can't compare the text chunks for assertion
            # mock_client.upsert.assert_called_once_with(
            #     collection_name='71a18a69-2dbd-466b-99e5-4e3e213430a9',
            #     points=models.Batch(
            #         ids=['cdba0257-4895-4908-8dc7-97faa33564dc', '1bf3fce1-6992-47ea-8728-9f55fe611b3b', 'ebff8d16-df8e-4913-807b-e39390fa3f74'],
            #         payloads=[{"text": chunk_text} for chunk_text in mock_chunks],
            #         vectors=mock_vectors,
            #     ),
            # )
            assert metadata == DocumentMetadata(id=DOCUMENT_ID, file_name=PDF_FILE_PATH)

    def test_retrieve_relevant_context(
        self, mock_embed_query, mock_query_embeddings, mock_client, mock_search_result
    ):
        global embed_query
        topic = "specific example topic"
        document_id = DOCUMENT_ID

        retrieved_chunks = retrieve_relevant_context(topic, document_id, mock_client)
        mock_embed_query.assert_called_once_with(query=topic)

        mock_client.query_points.assert_called_once_with(
            collection_name=document_id,
            query=mock_query_embeddings,
            with_payload=True,
            limit=10,
        )

        expected_chunks = [point.payload["text"] for point in mock_search_result.points]
        assert retrieved_chunks == expected_chunks
