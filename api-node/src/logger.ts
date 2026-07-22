import { AsyncLocalStorage } from 'async_hooks';

export interface RequestContext {
  requestId: string;
  apiKey?: string;
}

// Instance of AsyncLocalStorage to hold request-scoped information
export const contextStorage = new AsyncLocalStorage<RequestContext>();

/**
 * Log an informational message with structured JSON formatting.
 */
export function logInfo(message: string, extra: Record<string, any> = {}) {
  const store = contextStorage.getStore();
  console.log(JSON.stringify({
    timestamp: new Date().toISOString(),
    severity: 'INFO',
    message,
    request_id: store?.requestId || '',
    ...extra
  }));
}

/**
 * Log an error message with structured JSON formatting.
 */
export function logError(message: string, error?: any, extra: Record<string, any> = {}) {
  const store = contextStorage.getStore();
  console.error(JSON.stringify({
    timestamp: new Date().toISOString(),
    severity: 'ERROR',
    message,
    error: error?.message || String(error),
    stack: error?.stack || undefined,
    request_id: store?.requestId || '',
    ...extra
  }));
}
