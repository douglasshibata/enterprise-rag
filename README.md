# Enterprise Knowledge Search & RAG Platform

This project demonstrates a production-ready search and retrieval system using Python (FastAPI) for ETL, OpenSearch for hybrid search (BM25 + KNN), and Node.js (TypeScript) for RAG orchestration.

## Architecture

1.  **Ingestion & ETL (Python)**: Uses FastAPI, NLTK for chunking, and Sentence-Transformers for embedding generation.
2.  **Search Engine (OpenSearch)**: Hybrid search combining lexical (BM25) and semantic (KNN) retrieval.
3.  **Search API (Node.js/TypeScript)**: Exposes endpoints for search and RAG-style Q&A.

## Project Structure

- `backend-python/`: FastAPI service for document ingestion and embedding.
- `api-node/`: Node.js/TypeScript service for search and RAG orchestration.
- `docker-compose.yml`: Orchestrates OpenSearch and the two microservices.

## Getting Started

### Prerequisites

- Docker and Docker Compose
- Node.js (for local development)
- Python 3.12 (for local development)

### Running with Docker

```bash
docker compose up -d
```

### API Endpoints

- **Python Ingestion**: `POST http://localhost:8001/ingest`
  ```json
  {
    "text": "Your document text here...",
    "source": "manual-upload"
  }
  ```
- **Node.js Search**: `GET http://localhost:3001/search?q=your+query`
- **Node.js RAG**: `POST http://localhost:3001/ask`
  ```json
  {
    "question": "What is the document about?"
  }
  ```

## Development

### Python Backend
```bash
cd backend-python
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Node.js API
```bash
cd api-node
npm install
npm run dev
```
