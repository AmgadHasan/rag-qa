from pydantic import BaseModel

class DocumentMetadata(BaseModel):
    id: str
    file_name: str

class QA(BaseModel):
    length: int
    text: str

class Summary(BaseModel):
    text: str
    length: int