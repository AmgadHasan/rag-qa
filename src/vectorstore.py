from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
import pymupdf
from models import DocumentMetadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding import embed_texts, embed_query, EMBEDDING_DIMENSION
import os
import io
import uuid
from utils import get_logger

logger = get_logger(logger_name="vectorstore", log_file="api.log", log_level='info')
# Configuration for Qdrant client
QDRANT_HOST = os.environ.get("QDRANT_HOST", "http://localhost")
QDRANT_PORT = os.environ.get("QDRANT_PORT", 6333)
client = QdrantClient(url=f"{QDRANT_HOST}:{QDRANT_PORT}")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

def load_and_split_document(document_file: io.BytesIO) -> list[str]:
    """
    Load a PDF document from a byte stream and split it into text chunks.

    Args:
        document_file (io.BytesIO): The PDF document as a byte stream.

    Returns:
        list[str]: A list of text chunks.
    """
    doc = pymupdf.Document(stream=document_file, filetype='pdf')
    document_text = ""
    for page in doc:
        document_text += page.get_text()
    
    chunks = text_splitter.create_documents([document_text])
    chunks_texts = [chunk.page_content for chunk in chunks]
    
    logger.debug("Lengths after chunking")
    logger.debug(len(chunks_texts))
    
    return chunks_texts

def ingest_document(pdf_file: io.BytesIO) -> DocumentMetadata:
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
    except Exception as e:
        # Need proper error handling and logging here
        logger.error(e)
    
    return DocumentMetadata(id=collection_id, file_name=pdf_file.name)