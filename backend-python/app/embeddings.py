from sentence_transformers import SentenceTransformer

# Load a lightweight model
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(texts: list[str]):
    """
    Generates embeddings for a list of texts.
    :param texts: List of strings.
    :return: List of embeddings (vectors).
    """
    if not texts:
        return []
    embeddings = model.encode(texts)
    return embeddings.tolist()

def generate_query_embedding(query: str):
    """
    Generates an embedding for a single query.
    :param query: The search query string.
    :return: The embedding vector.
    """
    embedding = model.encode(query)
    return embedding.tolist()
