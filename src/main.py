import traceback

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from qdrant_client import QdrantClient

from src import models
from src.llm import provide_questions, summarize_topic
from src.utils import create_logger, process_uploaded_file, validate_pdf_file
from src.vectorstore import (
    get_qdrant_client,
    ingest_document,
    retrieve_relevant_context,
)

logger = create_logger(logger_name="main", log_file="api.log", log_level="info")

app = FastAPI()


@app.get("/")
async def root() -> dict:
    """
    Root endpoint that returns a welcome message.

    Returns:
        dict: A dictionary containing a welcome message.
    """
    return {"message": "Hello World"}


@app.post("/ingest")
async def ingest_pdf(
    file: UploadFile = File(...), client: QdrantClient = Depends(get_qdrant_client)
) -> models.DocumentMetadata:
    """
    Endpoint to ingest a PDF file into the vector store.

    Args:
        file (UploadFile): The PDF file to be ingested.
        client (QdrantClient): An instance of the Qdrant client.

    Returns:
        models.DocumentMetadata: Metadata of the ingested document.

    Raises:
        HTTPException: If there is an error processing the PDF.
    """
    validate_pdf_file(file)
    try:
        processed_pdf = await process_uploaded_file(file=file)
        document_metadata = ingest_document(pdf_file=processed_pdf, client=client)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    logger.info(f"Successfully ingested document: {document_metadata}")
    return document_metadata


@app.post("/generate/summary")
async def generate_summary(
    request: models.SummaryRequest, client: QdrantClient = Depends(get_qdrant_client)
) -> models.SummaryResponse:
    """
    Endpoint to generate a summary of a topic from a document.

    Args:
        request (models.SummaryRequest): The request containing the topic and document ID.

    Returns:
        models.SummaryResponse: The summary of the topic.

    Raises:
        HTTPException: If there is an error generating the summary.
    """
    try:
        chunks = retrieve_relevant_context(
            topic=request.topic, document_id=request.document_id, client=client
        )
        summary = summarize_topic(topic=request.topic, relevant_chunks=chunks)
        return models.SummaryResponse(topic=request.topic, summary=summary)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error generating summary: {str(e)}"
        )


@app.post("/generate/questions")
async def generate_questions(
    request: models.QuestionsRequest, client: QdrantClient = Depends(get_qdrant_client)
) -> models.QuestionsResponse:
    """
    Endpoint to generate questions about a topic from a document.

    Args:
        request (models.QuestionsRequest): The request containing the topic, document ID, and question type.

    Returns:
        models.QuestionsResponse: The generated questions.

    Raises:
        HTTPException: If there is an error generating the questions.
    """
    try:
        chunks = retrieve_relevant_context(
            topic=request.topic, document_id=request.document_id, client=client
        )
        questions = provide_questions(
            topic=request.topic, type=request.questions_type, relevant_chunks=chunks
        )
        return models.QuestionsResponse(
            topic=request.topic,
            questions_type=request.questions_type,
            questions=questions,
        )
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500, detail=f"Error generating questions: {str(e)}"
        )
