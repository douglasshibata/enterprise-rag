import express, { Request, Response } from 'express';
import { hybridSearch } from './search';
import { askQuestion } from './rag';
import dotenv from 'dotenv';

dotenv.config();

const app = express();
const port = process.env.PORT || 3000;

app.use(express.json());

app.get('/search', async (req: Request, res: Response) => {
  const query = req.query.q as string;
  if (!query) {
    return res.status(400).json({ error: 'Query parameter "q" is required' });
  }

  try {
    const results = await hybridSearch(query);
    res.json(results);
  } catch (error: any) {
    console.error('Search error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.post('/ask', async (req: Request, res: Response) => {
  const { question } = req.body;
  if (!question) {
    return res.status(400).json({ error: 'Field "question" is required' });
  }

  try {
    const response = await askQuestion(question);
    res.json(response);
  } catch (error: any) {
    console.error('RAG error:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/health', (req: Request, res: Response) => {
  res.json({ status: 'ok' });
});

app.listen(port, () => {
  console.log(`Node.js Search API listening at http://localhost:${port}`);
});
