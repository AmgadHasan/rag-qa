import traceback

from fastapi import FastAPI, File, HTTPException, UploadFile, Depends

from src import models
from src.utils import create_logger, process_uploaded_file, validate_pdf_file
from src.vectorstore import ingest_document, retrieve_relevant_context, get_qdrant_client
from src.llm import summarize_topic
from qdrant_client import QdrantClient

logger = create_logger(logger_name="main", log_file="api.log", log_level='info')
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...), client: QdrantClient = Depends(get_qdrant_client)) -> models.DocumentMetadata:
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
async def generate_summary(topic: str, document_id: str, client: QdrantClient = Depends(get_qdrant_client)) -> models.SummaryResponse:
    try:
        chunks = retrieve_relevant_context(topic=topic, document_id=document_id, client=client)
        summary = summarize_topic(topic=topic, relevant_chunks=chunks)
        return models.SummaryResponse(topic=topic, summary=summary)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")