import uuid
import logging
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Response, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from contextlib import asynccontextmanager

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import Counter, Histogram, generate_latest

from app.logging_config import setup_logging, request_id_var
from app.config import settings
from app.chunking import chunk_text
from app.embeddings import generate_embeddings, generate_query_embedding
from app.opensearch_client import OpenSearchClient
from app.ingest import fetch_url_content
from dotenv import load_dotenv

load_dotenv()

# Initialize structured logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
REQUEST_DURATION = Histogram("http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"])

os_client = OpenSearchClient()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create index
    try:
        os_client.create_index()
    except Exception as e:
        logger.error(f"Failed to create OpenSearch index at startup: {e}")
    yield
    # Shutdown logic (if any)

app = FastAPI(title="Enterprise Search Ingestion Service", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Custom Exception Handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

# Request ID & Tracing Middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    # Read custom request header or generate a new UUID
    req_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    token = request_id_var.set(req_id)
    try:
        request.state.request_id = req_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        return response
    finally:
        request_id_var.reset(token)

# Prometheus Metrics Middleware
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    if request.url.path in ["/metrics", "/health"]:
        return await call_next(request)

    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_DURATION.labels(method=request.method, endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()

    return response

# Security: Basic API Key authentication
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if not api_key or api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

class IngestRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    source: str
    chunk_size: Optional[int] = 5
    overlap: Optional[int] = 1

class EmbedRequest(BaseModel):
    text: str

@app.post("/embed")
@limiter.limit("30/minute")
async def embed_text(request: Request, embed_request: EmbedRequest, _api_key: str = Depends(get_api_key)):
    try:
        embedding = generate_query_embedding(embed_request.text)
        return {"embedding": embedding}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Embedding failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
@limiter.limit("10/minute")
async def ingest_document(request: Request, ingest_request: IngestRequest, _api_key: str = Depends(get_api_key)):
    try:
        content = ingest_request.text
        if ingest_request.url:
            logger.info(f"Fetching content from URL: {ingest_request.url}")
            content = fetch_url_content(ingest_request.url)

        if not content:
            raise HTTPException(status_code=400, detail="Either 'text' or 'url' must be provided.")

        # 1. Chunking
        chunks = chunk_text(content, ingest_request.chunk_size, ingest_request.overlap)
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
                "source": ingest_request.source
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
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    try:
        os_client.client.info()
        opensearch_ok = True
    except Exception as e:
        logger.error(f"OpenSearch connection health check failed: {e}")
        opensearch_ok = False

    status = "healthy" if opensearch_ok else "unhealthy"
    return {
        "status": status,
        "opensearch": opensearch_ok
    }

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")
