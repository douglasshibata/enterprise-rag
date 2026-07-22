import sqlite3
import os
import logging
import datetime

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("SQLITE_DB_PATH", "documents.db")

def init_db():
    """
    Initializes the SQLite database and creates the documents table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documentos (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                topic TEXT NOT NULL,
                full_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        logger.info(f"Initialized SQLite database at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {e}")
        raise e
    finally:
        conn.close()

def save_document_copy(doc_id: str, filename: str, topic: str, full_text: str):
    """
    Saves a copy of the processed document to the relational SQLite database.
    """
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO documentos (doc_id, filename, topic, full_text, created_at) VALUES (?, ?, ?, ?, ?)",
            (doc_id, filename, topic, full_text, created_at)
        )
        conn.commit()
        logger.info(f"Successfully saved document {doc_id} to SQLite database")
    except Exception as e:
        logger.error(f"Failed to save document {doc_id} to SQLite database: {e}")
        raise e
    finally:
        conn.close()
