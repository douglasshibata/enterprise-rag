import { Client } from '@opensearch-project/opensearch';
import axios from 'axios';
import dotenv from 'dotenv';
import Redis from 'ioredis';

dotenv.config();

const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');

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
 * Calls the Python backend to generate an embedding for the query,
 * with a Redis caching layer for performance.
 */
async function getQueryEmbedding(query: string): Promise<number[]> {
  const cacheKey = `embedding:${Buffer.from(query).toString('base64').substring(0, 32)}`;

  try {
    const cached = await redis.get(cacheKey);
    if (cached) {
      console.log('Using cached embedding for query');
      return JSON.parse(cached);
    }
  } catch (err) {
    console.error('Redis cache error:', err);
  }

  const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';
  const response = await axios.post(`${pythonBackendUrl}/embed`, { text: query });
  const embedding = response.data.embedding;

  try {
    // Cache for 1 hour
    await redis.set(cacheKey, JSON.stringify(embedding), 'EX', 3600);
  } catch (err) {
    console.error('Failed to cache embedding:', err);
  }

  return embedding;
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
