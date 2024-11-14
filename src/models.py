from pydantic import BaseModel

class DocumentMetadata(BaseModel):
    id: str
    file_name: str

class QA(BaseModel):
    topic: str
    type: str
    text: str

class SummaryResponse(BaseModel):
    topic: str
    summary: str