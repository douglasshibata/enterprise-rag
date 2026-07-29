import uuid
import logging
import time
import os
import re
import shutil
import tempfile
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException, Request, Response, Depends, Security, UploadFile, File, Form
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
from app.ingest import fetch_url_content, ingestar_documento_no_topico
from app.guardrails import check_input_guardrail
from app.database import init_db
from dotenv import load_dotenv

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

load_dotenv()

# Initialize structured logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize metrics
from prometheus_client import REGISTRY
if "http_requests_total" in REGISTRY._names_to_collectors:
    try:
        REGISTRY.unregister(REGISTRY._names_to_collectors["http_requests_total"])
    except KeyError:
        pass
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])

if "http_request_duration_seconds" in REGISTRY._names_to_collectors:
    try:
        REGISTRY.unregister(REGISTRY._names_to_collectors["http_request_duration_seconds"])
    except KeyError:
        pass
REQUEST_DURATION = Histogram("http_request_duration_seconds", "HTTP request duration", ["method", "endpoint"])

os_client = OpenSearchClient()

class MockOrRealLLM(LLM):
    model_name: str = "mock-llama3"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        # Check if we should use a real LLM (like OpenAI if configured)
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key != "mock-key":
            try:
                from langchain_openai import ChatOpenAI
                real_llm = ChatOpenAI(api_key=openai_key)
                return real_llm.predict(prompt)
            except Exception as e:
                logger.warning(f"Failed to use real OpenAI LLM: {e}")

        # Fallback simulated response
        context_match = re.search(r"Contexto:\s*(.*?)\s*Pergunta:", prompt, re.DOTALL | re.IGNORECASE)
        context_str = context_match.group(1).strip() if context_match else ""

        if not context_str or "não possui dados suficientes" in context_str.lower() or context_str == "Não possui dados suficientes.":
            return "Não possuo dados suficientes no contexto fornecido para responder à sua pergunta."

        first_line = context_str.split("\n")[0] if context_str else ""
        return f"[MOCK LLM RESPONSE] Com base no contexto técnico fornecido (especificamente: {first_line[:150]}...), " \
               f"a rotação de logs e gerenciamento do sistema funcionam conforme documentado."

    @property
    def _llm_type(self) -> str:
        return "mock_or_real"

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

class QueryRequest(BaseModel):
    pergunta: str
    topico: str

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

@app.post("/api/v1/upload")
async def upload_pdf(
    topico: str = Form(...),
    file: UploadFile = File(...)
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos")

    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        chunks_count, doc_id = ingestar_documento_no_topico(temp_path, topico)
        return {
            "status": "sucesso",
            "mensagem": f"Arquivo '{file.filename}' indexado com sucesso!",
            "topico": topico,
            "doc_id": doc_id,
            "chunks_criados": chunks_count
        }
    except Exception as e:
        logger.error(f"Erro ao processar upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@app.post("/api/v1/chat")
async def chat_multitopic(query_request: QueryRequest):
    try:
        # 1. Active input guardrail
        pergunta_validada = check_input_guardrail(query_request.pergunta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        # 2. OpenSearch Vector Store & Retriever
        from app.ingest import SentenceTransformerEmbeddings, sentence_transformer_model
        embeddings = SentenceTransformerEmbeddings(sentence_transformer_model)

        host = os.getenv("OPENSEARCH_HOST", "localhost")
        port = int(os.getenv("OPENSEARCH_PORT", 9200))
        user = os.getenv("OPENSEARCH_USER", "admin")
        pass_ = os.getenv("OPENSEARCH_PASS", "admin")
        opensearch_url = f"http://{host}:{port}"

        vector_store = OpenSearchVectorSearch(
            index_name="documentos-multitopicos",
            embedding_function=embeddings,
            opensearch_url=opensearch_url,
            http_auth=(user, pass_),
        )

        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 3,
                "pre_filter": {
                    "bool": {
                        "filter": [
                            {"term": {"metadata.topico.keyword": query_request.topico}}
                        ]
                    }
                }
            }
        )

        # 3. Retrieve relevant documents
        try:
            docs = retriever.invoke(pergunta_validada)
        except Exception as e:
            logger.warning(f"Error invoking OpenSearch retriever: {e}")
            docs = []

        # 4. Create prompt template
        template = """Você é um assistente virtual especialista e preciso. Responda à pergunta do usuário utilizando APENAS as informações fornecidas no CONTEXTO abaixo.
Se a resposta não estiver contida no contexto, diga claramente "Não encontrei essa informação nos documentos fornecidos." e não tente inventar.

---
Contexto:
{context}
---

Pergunta:
{question}

RESPOSTA:"""
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            if not docs:
                return "Não possui dados suficientes."
            return "\n\n".join(f"[Chunk {i+1}]: {doc.page_content}" for i, doc in enumerate(docs))

        llm = MockOrRealLLM()
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        resposta = rag_chain.invoke(pergunta_validada)

        return {
            "topico_pesquisado": query_request.topico,
            "resposta": resposta,
            "documentos_relevantes": [
                {
                    "conteudo": doc.page_content,
                    "fonte": doc.metadata.get("nome_arquivo", "Desconhecido")
                } for doc in docs
            ]
        }
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
