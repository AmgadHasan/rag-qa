import json
import os

from openai import OpenAI

from src.models import QuestionsType
from src.prompts import (
    QUESTIONS_SYSTEM_MESSAGE,
    QUESTIONS_USER_MESSAGE,
    SUMMARY_SYSTEM_MESSAGE,
    SUMMARY_USER_MESSAGE,
)
from src.utils import create_logger, log_execution_time

TEMPERATURE = 0.2
MAX_COMPLETION_TOKENS = 1024

logger = create_logger(logger_name="llm", log_file="api.log", log_level="info")

model = os.environ.get("CHAT_MODEL")
if not model:
    logger.error("CHAT_MODEL environment variable is not set.")
    raise ValueError("CHAT_MODEL environment variable is not set.")

client = OpenAI()


def prepare_context(relevant_chunks: list[str]) -> str:
    """
    Prepare the input context for the summarization model.
    Args:
        relevant_chunks (List[str]): A list of text chunks relevant to the topic.
    Returns:
        str: The concatenated text of the relevant chunks.
    """
    # This can be improved for prompt engineering!
    return "\n".join(relevant_chunks)


@log_execution_time(logger=logger)
def provide_questions(
    topic: str, type: QuestionsType, relevant_chunks: list[str]
) -> list:
    """
    Generate questions for a given topic based on relevant chunks of text.

    Args:
        topic (str): The main topic to be asked about.
        type( Enum["MCQ", "fill-in-the-middle"]): The type of questions to generate
        relevant_chunks (list[str]): A list of text chunks relevant to the topic.

    Returns:
        list[str]: A list of questions about the topic based on the provided context.
    """
    context = prepare_context(relevant_chunks)
    logger.debug(f"topic:\t{topic}\n\ncontext:\n{context}")
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": QUESTIONS_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": QUESTIONS_USER_MESSAGE.format(
                        topic=topic, type=type, context=context
                    ),
                },
            ],
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_COMPLETION_TOKENS,
            response_format={"type": "json_object"},
        )
        text_response = completion.choices[0].message.content
        print(f"text_response:\n{text_response}")
        questions = json.loads(text_response)["questions"]
    except Exception as e:
        logger.error(f"Error generating questions for topic '{topic}': {e}")
        raise
    return questions


@log_execution_time(logger=logger)
def summarize_topic(topic: str, relevant_chunks: list[str]) -> str:
    """
    Generate a summary for a given topic based on relevant chunks of text.

    Args:
        topic (str): The main topic to be summarized.
        relevant_chunks (list[str]): A list of text chunks relevant to the topic.

    Returns:
        str: A summary of the topic based on the provided context.
    """
    context = prepare_context(relevant_chunks)
    logger.debug(f"topic:\t{topic}\n\ncontext:\n{context}")
    try:
        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": SUMMARY_SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": SUMMARY_USER_MESSAGE.format(
                        topic=topic, context=context
                    ),
                },
            ],
            model=model,
            temperature=TEMPERATURE,
            max_tokens=MAX_COMPLETION_TOKENS,
        )
        response = completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating summary for topic '{topic}': {e}")
        raise
    return response
