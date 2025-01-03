import io
import logging
import time

from fastapi import HTTPException
from starlette.datastructures import UploadFile


def create_logger(logger_name, log_file, log_level):
    LOG_FORMAT = "[%(asctime)s | %(name)s | %(levelname)s | %(message)s]"
    log_level = getattr(logging, log_level.upper())

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)

    return logger


def log_execution_time(logger):
    """Decorator factory to log the execution time of a function using a specified logger."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Executing {func.__name__} took {execution_time:.4f} seconds")
            return result

        return wrapper

    return decorator


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

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Invalid file format. Only PDF files are allowed"
        )
