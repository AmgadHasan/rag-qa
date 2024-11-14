from enum import Enum

from pydantic import BaseModel


class QuestionsType(str, Enum):
    MCQ = "MCQ"
    FillInTheBlank = "fill-in-the-blank"


class DocumentMetadata(BaseModel):
    id: str
    file_name: str


class QuestionsRequest(BaseModel):
    topic: str
    document_id: str
    questions_type: QuestionsType

    class Config:
        use_enum_values = True


class QuestionsResponse(BaseModel):
    topic: str
    questions_type: QuestionsType
    questions: list

    class Config:
        use_enum_values = True


class SummaryRequest(BaseModel):
    topic: str
    summary: str


class SummaryResponse(BaseModel):
    topic: str
    summary: str
