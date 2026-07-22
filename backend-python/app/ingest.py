import os
import datetime
import requests
from bs4 import BeautifulSoup
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_core.embeddings import Embeddings
from app.embeddings import model as sentence_transformer_model

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self.model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(text).tolist()

def ingestar_documento_no_topico(caminho_pdf: str, topico_escolhido: str, host: str = None, port: int = None, user: str = None, pass_: str = None):
    """
    Loads, chunks and indexes a PDF document into the 'documentos-multitopicos' index under the specified topic.
    """
    # 1. Load PDF
    loader = PyPDFLoader(caminho_pdf)
    documents = loader.load()

    # 2. Chunk text with RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)

    # 3. Inject metadata into each chunk
    nome_arquivo = os.path.basename(caminho_pdf)
    data_upload = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for chunk in chunks:
        chunk.metadata["topico"] = topico_escolhido
        chunk.metadata["nome_arquivo"] = nome_arquivo
        chunk.metadata["data_upload"] = data_upload

    # 4. Setup embeddings wrapper
    embeddings = SentenceTransformerEmbeddings(sentence_transformer_model)

    # 5. Connection parameters
    if not host:
        host = os.getenv("OPENSEARCH_HOST", "localhost")
    if not port:
        port = int(os.getenv("OPENSEARCH_PORT", 9200))
    if not user:
        user = os.getenv("OPENSEARCH_USER", "admin")
    if not pass_:
        pass_ = os.getenv("OPENSEARCH_PASS", "admin")

    opensearch_url = f"http://{host}:{port}"

    # 6. Bulk ingest to OpenSearch
    vector_store = OpenSearchVectorSearch.from_documents(
        documents=chunks,
        embedding=embeddings,
        opensearch_url=opensearch_url,
        http_auth=(user, pass_),
        index_name="documentos-multitopicos",
        use_ssl=False,
        verify_certs=False,
    )

    return len(chunks)

def fetch_url_content(url: str) -> str:
    """
    Fetches the content of a URL and extracts the text.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()

        # Get text
        text = soup.get_text()

        # Break into lines and remove leading and trailing whitespace
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)

        return text
    except Exception as e:
        logger.error(f"Failed to fetch URL {url}: {str(e)}")
        raise e
