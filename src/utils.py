import io
from starlette.datastructures import UploadFile

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