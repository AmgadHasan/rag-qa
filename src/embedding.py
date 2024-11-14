import os

import requests

from src.utils import create_logger, log_execution_time

# Environment variables
JINA_API_KEY = os.environ.get("JINA_API_KEY")
MODEL = os.environ.get("JINA_MODEL", "jina-embeddings-v3")
EMBEDDING_DIMENSION = 1024

# API endpoint
URL = "https://api.jina.ai/v1/embeddings"

logger = create_logger(logger_name="embedding", log_file="api.log", log_level="info")

# HTTP headers for API requests
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {JINA_API_KEY}",
}


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
    embeddings = []
    for i in range(0, len(texts), 8):
        chunk = texts[i : i + 8]
        data = {
            "input": chunk,
            "model": MODEL,
            "dimensions": EMBEDDING_DIMENSION,
            "task": "retrieval.passage",
            "late_chunking": True,
        }
        response = requests.post(URL, headers=headers, json=data)
        chunk_embeddings = [d["embedding"] for d in response.json()["data"]]
        embeddings.extend(chunk_embeddings)

    return embeddings


@log_execution_time(logger=logger)
def embed_query(query: str) -> list[float]:
    """
    Get an embedding for a single query from the Jina API.

    Args:
        query (str): The query text to be embedded.

    Returns:
        list[float]: The embedding for the query, represented as a list of floats.
    """
    data = {
        "input": query,
        "model": MODEL,
        "dimensions": EMBEDDING_DIMENSION,
        "task": "retrieval.query",
        "late_chunking": True,
    }
    response = requests.post(URL, headers=headers, json=data)
    embeddings = [d["embedding"] for d in response.json()["data"]]
    return embeddings[0]
