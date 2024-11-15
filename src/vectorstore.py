import io
import os
import uuid

import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams

from src.emb import EMBEDDING_DIMENSION, embed_query, embed_texts
from src.models import DocumentMetadata
from src.utils import create_logger, log_execution_time

logger = create_logger(logger_name="vectorstore", log_file="api.log", log_level="info")


def get_qdrant_client() -> QdrantClient:
    QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=QDRANT_URL)
    try:
        yield client
    finally:
        client.close()


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)


def split_text(text):
    chunks = text_splitter.create_documents([text])
    chunks_texts = [chunk.page_content for chunk in chunks]

    return chunks_texts


def load_and_split_document(document_file: io.BytesIO) -> list[str]:
    """
    Load a PDF document from a byte stream and split it into text chunks.

    Args:
        document_file (io.BytesIO): The PDF document as a byte stream.

    Returns:
        list[str]: A list of text chunks.
    """
    doc = pymupdf.Document(stream=document_file, filetype="pdf")
    document_text = ""
    for page in doc:
        document_text += page.get_text()
    chunks = split_text(document_text)

    logger.debug(f"Lengths after chunking: {len(chunks)}")

    return chunks


def ingest_document(pdf_file: io.BytesIO, client: QdrantClient) -> DocumentMetadata:
    """
    Ingest a PDF document into Qdrant by creating a collection, splitting the document,
    embedding the text chunks, and upserting them into the collection.

    Args:
        pdf_file (io.BytesIO): The PDF document as a byte stream.

    Returns:
        DocumentMetadata: Metadata of the ingested document.
    """
    collection_id = str(uuid.uuid4())

    client.create_collection(
        collection_name=collection_id,
        vectors_config=VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.DOT),
    )

    chunks = load_and_split_document(document_file=pdf_file)
    vectors = embed_texts(texts=chunks)

    try:
        client.upsert(
            collection_name=collection_id,
            points=models.Batch(
                ids=[str(uuid.uuid4()) for _ in range(len(chunks))],
                payloads=[{"text": chunk} for chunk in chunks],
                vectors=vectors,
            ),
        )
        logger.info(f"Generated {len(chunks)} chunks and upserted into vectordb")
        return DocumentMetadata(id=collection_id, file_name=pdf_file.name)
    except Exception as e:
        # Need proper error handling and logging here
        logger.exception(f"failed to upsert: {e}")
        raise


@log_execution_time(logger=logger)
def retrieve_relevant_context(
    topic: str, document_id: str, client: QdrantClient
) -> list[str]:
    try:
        query_embeddings = embed_query(query=topic)
        search_result = client.query_points(
            collection_name=document_id,
            query=query_embeddings,
            with_payload=True,
            limit=10,
        ).points
        logger.debug(f"{search_result=}")
        retrieved_chunks = [point.payload["text"] for point in search_result]
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks.")
        return retrieved_chunks
    except Exception as e:
        logger.exception(f"failed to retrieve relecant context: {e}")
        raise
