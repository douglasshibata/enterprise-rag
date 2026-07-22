from opensearchpy import OpenSearch, helpers
import logging
from app.config import settings

logger = logging.getLogger(__name__)

class OpenSearchClient:
    def __init__(self):
        host = settings.opensearch_host
        port = settings.opensearch_port
        auth = (settings.opensearch_user, settings.opensearch_pass)

        self.client = OpenSearch(
            hosts=[{'host': host, 'port': port}],
            http_compress=True,
            http_auth=auth,
            use_ssl=False, # Set to True for production with SSL
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False
        )
        self.index_name = settings.opensearch_index

    def create_index(self):
        """
        Creates the KNN index if it doesn't exist.
        """
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 24
                            }
                        }
                    },
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "integer"},
                    "source": {"type": "keyword"}
                }
            }
        }

        if not self.client.indices.exists(index=self.index_name):
            self.client.indices.create(index=self.index_name, body=index_body)
            logger.info(f"Created index {self.index_name}")
        else:
            logger.info(f"Index {self.index_name} already exists")

    def create_multitopic_index(self, index_name="documentos-multitopicos"):
        """
        Creates the optimized HNSW multi-topic index if it doesn't exist.
        """
        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100
                }
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 384,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            }
                        }
                    },
                    "metadata": {
                        "properties": {
                            "topico": {
                                "type": "keyword"
                            },
                            "nome_arquivo": {
                                "type": "keyword"
                            },
                            "data_upload": {
                                "type": "date"
                            }
                        }
                    }
                }
            }
        }

        if not self.client.indices.exists(index=index_name):
            self.client.indices.create(index=index_name, body=index_body)
            logger.info(f"Created multi-topic index {index_name} with HNSW mapping")
        else:
            logger.info(f"Multi-topic index {index_name} already exists")

    def bulk_index(self, documents):
        """
        Indexes a list of documents in bulk.
        """
        actions = [
            {
                "_index": self.index_name,
                "_source": doc
            }
            for doc in documents
        ]
        success, failed = helpers.bulk(self.client, actions)
        return success, failed
