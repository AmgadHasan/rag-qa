from fastapi import FastAPI, File, UploadFile, HTTPException

from utils import process_uploaded_file
from vectorstore import ingest_document
import models
import traceback

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)) -> models.DocumentMetadata:
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    try:
        pdf_file = await process_uploaded_file(file=file)
        document_metadata = ingest_document(pdf_file=pdf_file)
        print("MEtadata")
        print(document_metadata)
        return document_metadata
    
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

# @app.post("/generate/summary")
# async def ingest_pdf() -> models.?:
#     
