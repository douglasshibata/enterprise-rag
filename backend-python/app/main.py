from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uuid
import logging
from contextlib import asynccontextmanager
from app.chunking import chunk_text
from app.embeddings import generate_embeddings, generate_query_embedding
from app.opensearch_client import OpenSearchClient
from app.ingest import fetch_url_content
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os_client = OpenSearchClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create index
    os_client.create_index()
    yield
    # Shutdown logic (if any)

app = FastAPI(title="Enterprise Search Ingestion Service", lifespan=lifespan)

class IngestRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    source: str
    chunk_size: Optional[int] = 5
    overlap: Optional[int] = 1

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed_text(request: EmbedRequest):
    try:
        embedding = generate_query_embedding(request.text)
        return {"embedding": embedding}
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
async def ingest_document(request: IngestRequest):
    try:
        content = request.text
        if request.url:
            logger.info(f"Fetching content from URL: {request.url}")
            content = fetch_url_content(request.url)

        if not content:
            raise HTTPException(status_code=400, detail="Either 'text' or 'url' must be provided.")

        # 1. Chunking
        chunks = chunk_text(content, request.chunk_size, request.overlap)
        if not chunks:
            return {"message": "No content to ingest."}

        # 2. Embedding generation
        embeddings = generate_embeddings(chunks)

        # 3. Prepare documents for indexing
        doc_id = str(uuid.uuid4())
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            documents.append({
                "text": chunk,
                "embedding": embedding,
                "doc_id": doc_id,
                "chunk_id": i,
                "source": request.source
            })

        # 4. Bulk indexing
        success, failed = os_client.bulk_index(documents)

        return {
            "message": "Ingestion completed",
            "doc_id": doc_id,
            "chunks_processed": len(chunks),
            "indexed_successfully": success,
            "failed": failed
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok"}
