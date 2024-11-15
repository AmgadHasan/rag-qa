import os

import requests

from src.utils import create_logger, log_execution_time

JINA_API_KEY = os.environ.get("JINA_API_KEY")
MODEL = os.environ.get("JINA_MODEL", "jina-embeddings-v3")
EMBEDDING_DIMENSION = 1024

URL = "https://api.jina.ai/v1/embeddings"

logger = create_logger(logger_name="embedding", log_file="api.log", log_level="info")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {JINA_API_KEY}",
}


def request_embeddings(
    input: str | list[str],
    task: str = "retrieval.passage",
    model: str = MODEL,
    dimensions: int = EMBEDDING_DIMENSION,
    late_chunking: bool = True,
):
    data = {
        "input": input,
        "model": MODEL,
        "dimensions": EMBEDDING_DIMENSION,
        "task": "retrieval.passage",
        "late_chunking": True,
    }
    try:
        response = requests.post(URL, headers=headers, json=data)
        return [d["embedding"] for d in response.json()["data"]]
    except Exception as e:
        logger.exception(f"Failed to generate embeddigns: {e}\nResponse: {response}")
        raise


@log_execution_time(logger=logger)
def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Get embeddings for a list of texts from the Jina API.

    This function implements a simple batching algorithm to avoid rate limiting.
    It sends requests in chunks of 8 texts at a time. This needs to be refactored
    to send concurrent requests for better performance.

    Args:
        texts (list[str]): A list of text strings to be embedded.

    Returns:
        list[list[float]]: A list of embeddings, where each embedding is a list of floats.
    """
    full_embeddings = []
    for i in range(0, len(texts), 8):
        batch = texts[i : i + 8]
        batch_embeddings = request_embeddings(input=batch, task="retrieval.passage")
        full_embeddings.extend(batch_embeddings)

    return full_embeddings


@log_execution_time(logger=logger)
def embed_query(query: str) -> list[float]:
    """
    Get an embedding for a single query from the Jina API.

    Args:
        query (str): The query text to be embedded.

    Returns:
        list[float]: The embedding for the query, represented as a list of floats.
    """
    batch_embeddings = request_embeddings(input=query, task="retrieval.query")
    return batch_embeddings[0]
