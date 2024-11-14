import io
from starlette.datastructures import UploadFile
import logging
from fastapi import HTTPException

def get_logger(logger_name, log_file, log_level):
    # Set up logging
    LOG_FORMAT = '[%(asctime)s | %(name)s | %(levelname)s | %(message)s]'
    log_level = getattr(logging, log_level.upper())  # convert to uppercase

    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # set the logger level to DEBUG

    # Add a file handler to log to a file with DEBUG level
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    # Add a console handler to log to the console with INFO level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    return logger

async def process_uploaded_file(file: UploadFile) -> io.BytesIO:
    """
    Process an uploaded file by reading its content and converting it into a BytesIO object.

    Args:
        file (UploadFile): The uploaded file object.

    Returns:
        io.BytesIO: A BytesIO object containing the file content.
    """
    file_content = await file.read()
    pdf_file = io.BytesIO(file_content)
    pdf_file.name = file.filename
    
    return pdf_file

def validate_pdf_file(file: UploadFile) -> None:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Only PDF files are allowed"
        )