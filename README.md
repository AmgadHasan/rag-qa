# rag-qa
Agent Framework for RAG-Based Question Generation and Summarization

## Author
Name: Amgad Hasan
LinkedIn: https://www.linkedin.com/in/amgad-hasan/
Website: https://amgadhasan.substack.com/

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

## Endpoints
Go to http://localhost:8000/docs and try out the different endpoints.

### 1. `/ingest`
Example request using `ML Engineer Tech Test.pdf` file:
```sh
curl -X 'POST' \
  'http://localhost:8000/ingest' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@./samples/markdown-file-02.pdf;type=application/pdf'
``` 
response:
```
{
  "id": "c085b661-5208-43cb-b045-da9ced4ed16a",
  "file_name": "markdown-file-02.pdf"
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
  "topic": "convolution",
  "document_id": "c085b661-5208-43cb-b045-da9ced4ed16a",
  "questions_type": "MCQ"
}'
``` 
response:
```json
{
  "topic": "convolution",
  "questions_type": "MCQ",
  "questions": "- Which of the following is NOT a key characteristic of convolutional layers in CNNs?\n    -  Feature extraction\n    -  Parameter sharing\n    -  Full connectivity\n    -  Local connectivity\n\n- What is the primary function of a pooling layer in a CNN?\n    -  Increase the dimensionality of feature maps\n    -  Introduce non-linearity into the network\n    -  Reduce the dimensionality of feature maps\n    -  Perform classification on the input data\n\n- What is the purpose of stride in a convolutional operation?\n    -  To control the receptive field of the convolutional filter\n    -  To adjust the number of output channels\n    -  To normalize the input data\n    -  To introduce randomness into the network\n\n- Which activation function is commonly used in the output layer of a CNN for multi-class classification?\n    -  ReLU\n    -  Sigmoid\n    -  Tanh\n    -  Softmax\n\n- What is the role of padding in a convolutional layer?\n    -  To prevent information loss at the edges of the input\n    -  To increase the computational complexity of the layer\n    -  To introduce noise into the input data\n    -  To normalize the output of the layer\n\n\n\n"
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
  "topic": "convolution",
  "document_id": "c085b661-5208-43cb-b045-da9ced4ed16a"
}'
``` 
response:
```json
{
  "topic": "convolution",
  "summary": "Convolutional Neural Networks (CNNs) are a specialized type of deep learning architecture designed primarily for processing grid-like data, such as images. \n\n**The core of CNNs lies in the \"convolutional layer.\"** This layer employs a set of learnable filters (also called kernels) that slide across the input data, performing element-wise multiplications and summings. This operation extracts features at different scales and locations within the input. \n\n**Think of it like this:** Imagine you're looking for a specific pattern in a picture. Instead of examining every pixel individually, you might use a magnifying glass (the filter) to scan the image and detect the presence of that pattern. CNNs use many filters to learn various features, from edges and corners to more complex shapes and textures.\n\n**Key benefits of convolutional layers:**\n\n* **Feature Extraction:** They automatically learn relevant features from the data, eliminating the need for manual feature engineering.\n* **Parameter Sharing:** Filters are applied across the entire input, sharing weights and reducing the number of parameters to learn.\n* **Translation Invariance:** CNNs are relatively robust to shifts and changes in the position of features within the input.\n\n**Beyond convolutional layers, CNNs typically include:**\n\n* **Pooling layers:** These downsample the feature maps, reducing dimensionality and making the network more robust to variations in input.\n* **Fully connected layers:** These layers perform classification or regression tasks based on the extracted features.\n\n**CNNs have revolutionized computer vision tasks like:**\n\n* Image classification\n* Object detection\n* Image segmentation\n* Facial recognition\n\nTheir success has also extended to other domains involving grid-like data, such as natural language processing (using word embeddings) and audio processing.\n\n\n"
}
```

## Development
### Testing
To run the tests, run:
```sh
uv run pytest
```

## Future work
These are some areas that could be improved upon:

### Async operations
We can improve the performance by implementing asnyc calls and making the code non-blocking in general.
### Optimizing Batch Embedding
Currently, we're looping sequentially over the chunks in minibatches of 8. We can improve this by implementing concurrent api requests.


### Structured Output for the LLM
In the questions generations endpoint, we can ask for a structured output of a certain schema to ensure proper parsing of the response.
