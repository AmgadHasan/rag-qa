from fastapi import FastAPI, File, UploadFile, HTTPException

from utils import process_uploaded_file, validate_pdf_file, get_logger
from vectorstore import ingest_document
import models
import traceback

logger = get_logger(logger_name="main", log_file="api.log", log_level='info')
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)) -> models.DocumentMetadata:
    validate_pdf_file(file)
    try:
        processed_pdf = await process_uploaded_file(file=file)
        document_metadata = ingest_document(pdf_file=processed_pdf)
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")
    logger.info(f"Successfully ingested document: {document_metadata}")
    return document_metadata

# @app.post("/generate/summary")
# async def ingest_pdf() -> models.?:
#     
