import { Client } from '@opensearch-project/opensearch';
import axios from 'axios';
import dotenv from 'dotenv';

dotenv.config();

const osHost = process.env.OPENSEARCH_HOST || 'localhost';
const osPort = process.env.OPENSEARCH_PORT || '9200';
const osProtocol = 'http';
const auth = `${process.env.OPENSEARCH_USER || 'admin'}:${process.env.OPENSEARCH_PASS || 'admin'}`;
const indexName = process.env.OPENSEARCH_INDEX || 'enterprise_docs';

const client = new Client({
  node: `${osProtocol}://${auth}@${osHost}:${osPort}`,
  ssl: {
    rejectUnauthorized: false,
  },
});

/**
 * Calls the Python backend to generate an embedding for the query.
 */
async function getQueryEmbedding(query: string): Promise<number[]> {
  const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
  // We need to implement this endpoint in Python or use the same logic here.
  // For simplicity, let's assume we add a /embed endpoint to the Python backend.
  const response = await axios.post(`${pythonBackendUrl}/embed`, { text: query });
  return response.data.embedding;
}

export async function hybridSearch(query: string, size: number = 5) {
  const queryEmbedding = await getQueryEmbedding(query);

  const body = {
    size: size,
    query: {
      bool: {
        should: [
          {
            match: {
              text: {
                query: query,
                boost: 1.0,
              },
            },
          },
          {
            knn: {
              embedding: {
                vector: queryEmbedding,
                k: size,
              },
            },
          },
        ],
      },
    },
  };

  const response = await client.search({
    index: indexName,
    body: body,
  });

  return response.body.hits;
}
