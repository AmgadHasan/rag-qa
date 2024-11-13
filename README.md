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

``` 
response:
```
{

}
```

### 3. `/generate/summary:`
Example request :
```sh
curl -X 'POST' \
  'http://localhost:8000/generate/summary' \
  -H 'accept: application/json' \

``` 
response:
```
{
    
}
```

## Draft
Requirements:
1. agent-based framework
2. Rag
3. PDF files
4. python
5. LLMs
6. Need instructions and documentations
7. 