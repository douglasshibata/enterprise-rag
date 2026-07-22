import os
import re
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import List, Optional, Any
import uuid
import logging
from contextlib import asynccontextmanager
from app.chunking import chunk_text
from app.embeddings import generate_embeddings, generate_query_embedding
from app.opensearch_client import OpenSearchClient
from app.ingest import fetch_url_content, ingestar_documento_no_topico
from app.guardrails import check_input_guardrail
from dotenv import load_dotenv

from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        logger.warning(f"Could not create default index on startup: {e}")

    try:
        os_client.create_multitopic_index()
    except Exception as e:
        logger.warning(f"Could not create multi-topic index on startup: {e}")
    yield
    # Shutdown logic (if any)

app = FastAPI(title="Enterprise Search Ingestion Service", lifespan=lifespan)

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

@app.post("/api/v1/upload")
async def upload_arquivo(
    topico: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Receives a PDF file and indexes its chunks under the specified topic.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são aceitos.")

    caminho_temp = f"/tmp/{uuid.uuid4()}_{file.filename}"

    # Save file locally
    try:
        with open(caminho_temp, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call ingestion helper
        total_chunks = ingestar_documento_no_topico(caminho_temp, topico)

        return {
            "status": "sucesso",
            "mensagem": f"Arquivo '{file.filename}' indexado com sucesso!",
            "topico": topico,
            "chunks_criados": total_chunks
        }
    except Exception as e:
        logger.error(f"Erro no upload/ingestão: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(caminho_temp):
            os.remove(caminho_temp)

@app.post("/api/v1/chat")
async def consultar_documentos(request: QueryRequest):
    """
    RAG chat endpoint. Performs hybrid search pre-filtered by the specified topic and uses LangChain LCEL.
    """
    try:
        # 1. Input Guardrail
        check_input_guardrail(request.pergunta)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Get vector store connection
    host = os.getenv("OPENSEARCH_HOST", "localhost")
    port = int(os.getenv("OPENSEARCH_PORT", 9200))
    user = os.getenv("OPENSEARCH_USER", "admin")
    pass_ = os.getenv("OPENSEARCH_PASS", "admin")
    opensearch_url = f"http://{host}:{port}"

    from app.ingest import SentenceTransformerEmbeddings
    from app.embeddings import model as sentence_transformer_model
    embeddings = SentenceTransformerEmbeddings(sentence_transformer_model)

    try:
        vector_store = OpenSearchVectorSearch(
            opensearch_url=opensearch_url,
            index_name="documentos-multitopicos",
            embedding_function=embeddings,
            http_auth=(user, pass_),
            use_ssl=False,
            verify_certs=False,
        )

        # Pre-filter by topic
        filtro_topico = {
            "bool": {
                "filter": [
                    {
                        "term": {
                            "metadata.topico": request.topico
                        }
                    }
                ]
            }
        }

        # 3. Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "boolean_filter": filtro_topico
            }
        )

        # 4. Define formatting
        def format_docs(docs):
            if not docs:
                return "Não possui dados suficientes."
            return "\n\n".join(f"[Fonte: {doc.metadata.get('nome_arquivo', 'Desconhecido')}]\n{doc.page_content}" for doc in docs)

        # 5. Build prompt
        template = """Você é um assistente técnico especializado. Responda à pergunta do usuário baseando-se APENAS no contexto fornecido abaixo. Se a resposta não estiver no contexto, responda que não possui dados suficientes.

Contexto:
{context}

Pergunta:
{question}

Resposta concisa e fundamentada:"""

        prompt = ChatPromptTemplate.from_template(template)

        # 6. Initialize LLM wrapper
        llm = MockOrRealLLM()

        # 7. Construct LCEL Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # 8. Execute the chain
        resposta = rag_chain.invoke(request.pergunta)

        # 9. Also fetch documents directly to provide citations
        try:
            documentos_encontrados = retriever.invoke(request.pergunta)
        except Exception:
            documentos_encontrados = []

        return {
            "topico_pesquisado": request.topico,
            "resposta": resposta,
            "documentos_relevantes": [
                {
                    "conteudo": doc.page_content,
                    "fonte": doc.metadata.get("nome_arquivo", "Desconhecido")
                } for doc in documentos_encontrados
            ]
        }
    except Exception as e:
        logger.error(f"Erro no chat RAG: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
