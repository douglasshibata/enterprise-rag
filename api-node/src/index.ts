import express, { Request, Response, NextFunction } from 'express';
import crypto from 'crypto';
import { hybridSearch, redis, client as openSearchClient } from './search';
import { askQuestion } from './rag';
import { contextStorage, logInfo, logError } from './logger';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

// Middleware: Propagate unique request ID using AsyncLocalStorage
app.use((req: Request, res: Response, next: NextFunction) => {
  const requestId = (req.headers['x-request-id'] as string) || crypto.randomUUID();
  res.setHeader('X-Request-ID', requestId);

  contextStorage.run({ requestId }, () => {
    logInfo(`${req.method} ${req.url} - Request received`);
    next();
  });
});

// Middleware: Simple API Key validation
const verifyApiKey = (req: Request, res: Response, next: NextFunction) => {
  const expectedKey = process.env.API_KEY || 'default-secret-key';
  const apiKey = req.headers['x-api-key'] as string;

  if (!apiKey || apiKey !== expectedKey) {
    logError('Access Denied - Invalid or missing API Key');
    return res.status(403).json({ error: 'Forbidden: Invalid or missing API Key' });
  }
  next();
};

app.get('/search', verifyApiKey, async (req: Request, res: Response, next: NextFunction) => {
  const query = req.query.q as string;
  if (!query) {
    return res.status(400).json({ error: 'Query parameter "q" is required' });
  }

  try {
    const results = await hybridSearch(query);
    res.json(results);
  } catch (error: any) {
    next(error);
  }
});

app.post('/ask', verifyApiKey, async (req: Request, res: Response, next: NextFunction) => {
  const { question } = req.body;
  if (!question) {
    return res.status(400).json({ error: 'Field "question" is required' });
  }

  try {
    const response = await askQuestion(question);
    res.json(response);
  } catch (error: any) {
    next(error);
  }
});

app.get('/health', async (req: Request, res: Response) => {
  let redisOk = false;
  let opensearchOk = false;

  try {
    const pingRes = await redis.ping();
    redisOk = pingRes === 'PONG';
  } catch (err) {
    logError('Redis health check failed', err);
  }

  try {
    const osInfo = await openSearchClient.info();
    opensearchOk = !!osInfo;
  } catch (err) {
    logError('OpenSearch health check failed', err);
  }

  const status = redisOk && opensearchOk ? 'healthy' : 'unhealthy';
  res.status(status === 'healthy' ? 200 : 503).json({
    status,
    redis: redisOk,
    opensearch: opensearchOk,
  });
});

// Middleware: Standard Global Error Handler
app.use((err: any, req: Request, res: Response, next: NextFunction) => {
  logError('Unhandled Express exception caught', err);
  res.status(500).json({ error: 'Internal server error' });
});

app.listen(port, () => {
  console.log(`Node.js Search API listening at http://localhost:${port}`);
});
