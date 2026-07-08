import nltk
from nltk.tokenize import sent_tokenize
import logging

logger = logging.getLogger(__name__)

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except (LookupError, OSError):
    nltk.download('punkt')

try:
    # NLTK sometimes throws OSError if the directory structure is half-present
    nltk.data.find('tokenizers/punkt_tab')
except (LookupError, OSError):
    nltk.download('punkt_tab')

def chunk_text(text: str, chunk_size: int = 5, overlap: int = 1):
    """
    Splits text into chunks of sentences with overlap.
    :param text: The input text to chunk.
    :param chunk_size: Number of sentences per chunk.
    :param overlap: Number of overlapping sentences between chunks.
    :return: A list of string chunks.
    """
    try:
        sentences = sent_tokenize(text)
    except Exception as e:
        logger.error(f"Tokenization failed: {str(e)}")
        # Fallback to simple split if NLTK fails
        sentences = [s.strip() for s in text.split('.') if s.strip()]

    chunks = []

    if not sentences:
        return []

    i = 0
    while i < len(sentences):
        # Create a chunk of 'chunk_size' sentences
        chunk_sentences = sentences[i : i + chunk_size]
        chunks.append(" ".join(chunk_sentences))

        # Move forward by (chunk_size - overlap)
        i += (chunk_size - overlap)

        # Avoid infinite loop if overlap >= chunk_size
        if chunk_size <= overlap:
            i += 1

        # Stop if we've processed all sentences
        if i >= len(sentences) - overlap and i != 0:
            if i >= len(sentences):
                break

    return chunks
