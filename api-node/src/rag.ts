import { hybridSearch } from './search';
import OpenAI from 'openai';
import dotenv from 'dotenv';
import { logInfo, logError } from './logger';

dotenv.config();

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || 'mock-key',
  baseURL: process.env.LLM_BASE_URL || undefined, // Can point to Ollama or LiteLLM
});

/**
 * Orchestrates the RAG flow:
 * 1. Retrieve relevant chunks from OpenSearch.
 * 2. Construct context.
 * 3. Call LLM (real call if API key provided, otherwise mock).
 */
export async function askQuestion(question: string) {
  logInfo('Starting RAG flow for question', { question: question.substring(0, 50) });
  const searchResults = await hybridSearch(question, 3);
  const hits = searchResults.hits;

  if (hits.length === 0) {
    logInfo('No relevant chunks found in OpenSearch for question');
    return {
      answer: "I couldn't find any relevant information to answer your question.",
      citations: [],
    };
  }

  logInfo(`Found ${hits.length} relevant chunks inside OpenSearch`);

  const context = hits
    .map((hit: any) => hit._source.text)
    .join('\n\n');

  const citations = hits.map((hit: any) => ({
    doc_id: hit._source.doc_id,
    chunk_id: hit._source.chunk_id,
    source: hit._source.source,
    text: hit._source.text.substring(0, 100) + '...',
  }));

  if (process.env.OPENAI_API_KEY && process.env.OPENAI_API_KEY !== 'mock-key') {
    try {
      logInfo('Invoking OpenAI completion API');
      const response = await openai.chat.completions.create({
        model: process.env.LLM_MODEL || 'gpt-3.5-turbo',
        messages: [
          { role: 'system', content: 'You are a helpful assistant. Answer the question using ONLY the provided context.' },
          { role: 'user', content: `Context:\n${context}\n\nQuestion: ${question}` },
        ],
      });

      return {
        answer: response.choices[0].message.content,
        citations,
      };
    } catch (error: any) {
      logError('LLM Call failed, falling back to mock response', error);
    }
  }

  // Fallback Mock LLM Response
  logInfo('OpenAI key not configured or failed, returning Fallback Mock Response');
  const answer = `[MOCK LLM ANSWER based on ${hits.length} chunks]
Relevant context found in ${hits[0]._source.source}.
The provided text mentions: "${hits[0]._source.text.substring(0, 150)}..."`;

  return {
    answer,
    citations,
  };
}
