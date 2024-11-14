# rag-qa
Agent Framework for RAG-Based Question Generation and Summarization

## How to run
### 1. Prerequisites
Please make sure you have the following installed:
1. Docker Engine: https://docs.docker.com/engine/install/ubuntu/
2. Docker Compose: https://docs.docker.com/compose/install/linux/#install-using-the-repository

### 2. Set up the environments
Please create a `.env` file with by copying the `.example.env` file and populating needed fields:
```
cp .env.example .env
```
### 3. Run the containers
Start the application:
```sh
docker compose --profile dev up
```
or
```sh
bash run.sh
```
This will:
1. Start the qdrant vectorstore
2. Start the fastapi api server

## Using the framework
Go to http://localhost:8000/docs and try out the different endpoints.

## Endpoints
### 1. `/ingest`
Example request using `ML Engineer Tech Test.pdf` file:
```sh
curl -X 'POST' \
  'http://localhost:8000/ingest' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@ML Engineer Tech Test.pdf;type=application/pdf'
``` 
response:
```
{
  "id": "497f5241-2178-45c0-b5ef-c7a5e530db93",
  "file_name": "ML Engineer Tech Test.pdf"
}
```

### 2. `/generate/questions`
Example request :
```sh
curl -X 'POST' \
  'http://localhost:8000/generate/questions' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "topic": "slope",
  "document_id": "71a18a69-2dbd-466b-99e5-4e3e213430a9",
  "questions_type": "MCQ"
}'
``` 
response:
```json
{
  "topic": "slope",
  "questions_type": "MCQ",
  "questions": [
    "What does the slope of a line represent?",
    "What is the formula for calculating the slope of a line given two points?",
    "If the slope of a line is positive, what does this tell you about the line?",
    "If the slope of a line is zero, what does this tell you about the line?",
    "If the slope of a line is undefined, what does this tell you about the line?"
  ]
}
```

### 3. `/generate/summary:`
Example request :
```sh
curl -X 'POST' \
  'http://localhost:8000/generate/summary' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "topic": "slope",
  "document_id": "71a18a69-2dbd-466b-99e5-4e3e213430a9"
}'
``` 
response:
```json
{
  "detail": "Error generating summary: Error code: 400 - {'error': {'message': \"property 'max_completion_tokens' is unsupported, did you mean 'max_tokens'?\", 'type': 'invalid_request_error'}}"
}
```

## Future work
These are some areas that could be improved upon:

### Async operations
We can improve the performance by implementing asnyc calls and making the code non-blocking in general.
### Optimizing Batch Embedding
Currently, we're looping sequentially over the chunks in minibatches of 8. We can improve this by implementing concurrent api requests.


### Structured Output for the LLM
In the questions generations endpoint, we can ask for a structured output of a certain schema to ensure proper parsing of the response.