import { Client } from '@opensearch-project/opensearch';
import axios from 'axios';
import dotenv from 'dotenv';
import Redis from 'ioredis';
import { contextStorage, logInfo, logError } from './logger';

dotenv.config();

export const redis = new Redis(process.env.REDIS_URL || 'redis://localhost:6379');

const osHost = process.env.OPENSEARCH_HOST || 'localhost';
const osPort = process.env.OPENSEARCH_PORT || '9200';
const osProtocol = 'http';
const auth = `${process.env.OPENSEARCH_USER || 'admin'}:${process.env.OPENSEARCH_PASS || 'admin'}`;
const indexName = process.env.OPENSEARCH_INDEX || 'enterprise_docs';

export const client = new Client({
  node: `${osProtocol}://${auth}@${osHost}:${osPort}`,
  ssl: {
    rejectUnauthorized: false,
  },
});

/**
 * Custom resilient POST helper utilizing exponential backoff retry policies for transient backend errors.
 */
async function axiosPostWithRetry(url: string, data: any, config: any, maxRetries = 3): Promise<any> {
  let lastError: any;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await axios.post(url, data, config);
    } catch (err: any) {
      lastError = err;
      // Identify if the error is transient: a network error or a 5xx status code
      const isTransient = !err.response || (err.response.status >= 500 && err.response.status <= 599);
      if (!isTransient) {
        throw err;
      }
      const delay = Math.pow(2, attempt) * 1000;
      logInfo(`Transient error on POST ${url}. Retrying (attempt ${attempt + 1}/${maxRetries}) in ${delay}ms...`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw lastError;
}

/**
 * Calls the Python backend to generate an embedding for the query,
 * with a Redis caching layer for performance.
 */
async function getQueryEmbedding(query: string): Promise<number[]> {
  const cacheKey = `embedding:${Buffer.from(query).toString('base64').substring(0, 32)}`;

  try {
    const cached = await redis.get(cacheKey);
    if (cached) {
      logInfo('Using cached embedding for query');
      return JSON.parse(cached);
    }
  } catch (err) {
    logError('Redis cache error:', err);
  }

  const pythonBackendUrl = process.env.PYTHON_BACKEND_URL || 'http://localhost:8000';

  // Set up request headers from the current context contextStorage
  const store = contextStorage.getStore();
  const headers: Record<string, string> = {};
  if (store?.requestId) {
    headers['X-Request-ID'] = store.requestId;
  }
  // If API Key is configured for python-backend, forward it
  const apiKey = process.env.API_KEY || 'default-secret-key';
  headers['X-API-Key'] = apiKey;

  logInfo('Generating embedding from Python backend service', { query: query.substring(0, 50) });

  const response = await axiosPostWithRetry(
    `${pythonBackendUrl}/embed`,
    { text: query },
    { headers }
  );

  const embedding = response.data.embedding;

  try {
    // Cache for 1 hour
    await redis.set(cacheKey, JSON.stringify(embedding), 'EX', 3600);
  } catch (err) {
    logError('Failed to cache embedding:', err);
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
