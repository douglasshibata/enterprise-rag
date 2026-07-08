# Enterprise Search & RAG Platform

Production-ready RAG orchestration platform featuring hybrid retrieval and scalable ingestion.

## 📋 Table of Contents
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Setup](#setup)
- [Security](#security)

## Architecture
- **ETL Pipeline (Python)**: Specialized FastAPI service for document chunking and embedding.
- **Hybrid Search (OpenSearch)**: BM25 lexical + KNN semantic retrieval.
- **RAG API (Node.js)**: TypeScript-based orchestration with LLM shielding.

## Tech Stack
- **Backend**: Python 3.12, Node.js 20+, TypeScript.
- **Search**: OpenSearch 2.8.0.
- **AI**: LangChain, sentence-transformers.

## Setup
1. `docker-compose up -d` (OpenSearch)
2. `cd backend-python && pip install -r requirements.txt && uvicorn main:app`
3. `cd api-node && npm install && npm run dev`

## Security
- **JSON Shielding**: Prevents prompt injection and ensures valid JSON responses.
- **Rate Limiting**: Integrated at the RAG API level.
- **Audit**: Log-level tracking of retrieval sources for transparency.

## License
Enterprise License
