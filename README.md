# Enterprise Search & RAG Platform

Production-ready, enterprise-grade multi-topic Retrieval-Augmented Generation (RAG) platform featuring hybrid retrieval, secure guardrails, and dual-layer ingestion.

## 📋 Table of Contents
- [Architecture & Design](#architecture--design)
- [Multi-Topic Single Index Strategy](#multi-topic-single-index-strategy-ram-optimized)
- [HNSW k-NN Index Configuration](#hnsw-k-nn-index-configuration)
- [Dual-Layer Storage Ingestion](#dual-layer-storage-ingestion)
- [Input Security Guardrails](#input-security-guardrails)
- [FastAPI Endpoints (`/api/v1`)](#fastapi-endpoints-apiv1)
- [Tech Stack](#tech-stack)
- [Setup & Run](#setup--run)
- [Tests](#tests)

---

## 🏗️ Architecture & Design

The platform divides operations into two primary flows:
* **Background/Offline Ingestion Pipeline**: Handles PDF uploads, extracts full text, chunks documents, inserts metadata, and indexes them into OpenSearch while persisting a copy in a relational database.
* **Online/Real-time Retrieval & Generation**: Uses LangChain LCEL (LangChain Expression Language) to construct a prompt context based on a pre-filtered topic search on OpenSearch and invokes the LLM.

```
 ┌────────────────┐     ┌───────────────────┐     ┌─────────────────────┐
 │  Documentos    │ ──> │ Chunking & Embed  │ ──> │ OpenSearch Index    │
 │ (PDF, MD, Doc) │     │ (LangChain Split) │     │ (BM25 + k-NN Vector)│
 └────────────────┘     └───────────────────┘     └─────────────────────┘
         │                                                   │
         ▼                                                   │ 1. Retrieval
 ┌────────────────┐     ┌───────────────────┐                │ (RRF / Hybrid)
 │ Relational DB  │     │ Busca Híbrida     │ <──────────────┘
 │ (SQLite Copy)  │     │ (OpenSearch)      │
 └────────────────┘     └───────────────────┘
                                  │
                                  │ 2. Contexto + Prompt
                                  v
                        ┌───────────────────┐     ┌─────────────────────┐
                        │ LangChain Chain   │ ──> │ Resposta Final da   │
                        │ + LLM (vLLM/Ollama│     │ LLM com Fontes      │
                        └───────────────────┘     └─────────────────────┘
```

---

## 🧠 Multi-Topic Single Index Strategy (RAM-Optimized)

Instead of creating an independent OpenSearch index per topic or user, this platform implements a **Single Index Architecture with Metadata Pre-Filtering** (`metadata.topico`).

### Why this is superior:
* **RAM Efficiency (Off-Heap)**: The OpenSearch HNSW navigation graphs are loaded directly into physical memory (Off-Heap Native Memory). Creating 100 separate indices results in 100 separate graphs, overwhelming the cluster's memory limit.
* **Unified Pipeline**: A single index allows easier scale-out, cleaner backup/snapshots, and unified shard tuning, while using boolean pre-filtering guarantees that search is restricted strictly to relevant documents of that specific topic.

---

## ⚙️ HNSW k-NN Index Configuration

The `documentos-multitopicos` index is explicitly configured with optimized Hierarchical Navigable Small World (HNSW) settings for high-accuracy and fast vector navigation:

| Configuration | Value | Description |
|---|---|---|
| `space_type` | `cosinesimil` | Cosine Similarity (recommended for text embeddings where vector direction is key). |
| `dimension` | `384` | Tailored to sentence-transformers/all-MiniLM-L6-v2. |
| `m` | `16` | Max bidirectional connections per node; perfectly balances memory consumption and recall. |
| `ef_construction` | `128` | Number of neighbors examined during index construction. |
| `ef_search` | `100` | Number of neighbors examined during search; yields high recall at minimal latency. |
| `metadata.topico` | `keyword` | Explicitly mapped as a keyword to guarantee O(1) exact match pre-filtering. |

---

## 🗄️ Dual-Layer Storage Ingestion

When a PDF file is submitted via the ingestion endpoint, the following workflow is executed:
* **Text Extraction**: The complete document text is extracted and reconstructed using LangChain's `PyPDFLoader`.
* **Relational Copy**: The extracted text is saved in a SQLite Relational Database with a strict schema `(doc_id PRIMARY KEY, filename, topic, full_text, created_at)` to keep a perfect record of the original content.
* **Chunking**: Text is split into 800-character chunks with a 150-character overlap using `RecursiveCharacterTextSplitter`.
* **Metadata Injection**: Each chunk inherits the generated `doc_id`, `topico`, and `nome_arquivo` as metadata.
* **Vector Ingestion**: Embeddings are generated using `sentence-transformers/all-MiniLM-L6-v2`, and both chunks and vector fields are indexed in OpenSearch.

---

## 🛡️ Input Security Guardrails

To prevent malicious instructions and LLM jailbreaks, the application implements active input shielding (`check_input_guardrail`):
* **Regex Filtering**: Instantly blocks suspect patterns such as "ignore previous instructions", "revelar seu prompt", "você agora é um", and "system prompt / sudo mode".
* **Size Enforcement**: Restricts prompt length to a maximum of 1000 characters, rejecting oversized payloads to prevent API billing exploitation and context overloading.

---

## 🔌 FastAPI Endpoints (`/api/v1`)

### 1. File Upload & Indexing (`POST /api/v1/upload`)
Receives a PDF file and registers its content under a designated topic.

| Property | Details |
|---|---|
| **Format** | `multipart/form-data` |
| **Params** | `topico` (string): e.g., "Financeiro"<br>`file` (`UploadFile`): PDF file |

**Success Response**:
```json
{
  "status": "sucesso",
  "mensagem": "Arquivo 'manual.pdf' indexado com sucesso!",
  "topico": "Financeiro",
  "doc_id": "87e9fa34-3199-4c12-a7d1-e970b86a87c1",
  "chunks_criados": 24
}
```

### 2. Multi-Topic RAG Chat (`POST /api/v1/chat`)
Answers a query using context retrieved only from documents matching the topic.

| Property | Details |
|---|---|
| **Format** | `application/json` |
| **Body** | `pergunta` (string): "Como configurar o backup?"<br>`topico` (string): "Infraestrutura" |

**Success Response**:
```json
{
  "topico_pesquisado": "Infraestrutura",
  "resposta": "A rotação de logs armazena arquivos compactados no diretório /var/log...",
  "documentos_relevantes": [
    {
      "conteudo": "A rotação de logs...",
      "fonte": "manual_logs.pdf"
    }
  ]
}
```

---

## 💻 Tech Stack

* **Backend Framework**: Python 3.12, FastAPI 0.115.8, Node.js 20+
* **LangChain Integration**: LangChain Core 0.3.x, LangChain Community, LCEL
* **Search Database**: OpenSearch 2.8.0
* **Relational Database**: SQLite
* **Embeddings**: Sentence-Transformers (`all-MiniLM-L6-v2`)

---

## 🚀 Setup & Run

### Docker Compose (All Services)
Start the OpenSearch node, Redis cache, Python backend, and Node.js orchestrator:
```bash
docker-compose up -d --build
```

### Run Python Backend Locally
1. Navigate to directory:
   ```bash
   cd backend-python
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install langchain langchain-community langchain-core pypdf python-multipart
   ```
3. Run FastAPI application:
   ```bash
   PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

---

## 🧪 Tests

The project includes an extensive unit and integration testing suite utilizing `pytest` to validate document ingestion logic, metadata injection, SQLite relational copy insertion, guardrail security filters, and REST API endpoints.

Execute tests:
```bash
PYTHONPATH=enterprise-rag/backend-python python3 -m pytest enterprise-rag/backend-python/tests/
```

---

> Enterprise Search & RAG Platform — Developed with high performance standards.
