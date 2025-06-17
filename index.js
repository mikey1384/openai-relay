import { createServer } from 'node:http';
import { fetch } from 'undici';

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const PORT = process.env.PORT || 3000;

// Enhanced relay for translator app - supports both transcription and translation
const server = createServer(async (req, res) => {
  // Enable CORS for all origins (restrict in production)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  if (req.method !== 'POST') {
    res.writeHead(405, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Method not allowed' }));
    return;
  }

  const url = new URL(req.url, 'http://localhost');
  let targetUrl;
  let contentType = req.headers['content-type'] || 'application/json';

  // Route to appropriate OpenAI endpoint
  if (url.pathname === '/transcribe') {
    targetUrl = 'https://api.openai.com/v1/audio/transcriptions';
  } else if (url.pathname === '/translate') {
    targetUrl = 'https://api.openai.com/v1/chat/completions';
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Endpoint not found' }));
    return;
  }

  // Validate OpenAI API key
  if (!OPENAI_API_KEY) {
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'OpenAI API key not configured' }));
    return;
  }

  let body;
  let abortController;

  try {
    // Set up cancellation handling
    abortController = new AbortController();
    
    // Cancel if client disconnects
    req.on('close', () => {
      console.log(`Client disconnected, aborting request to ${url.pathname}`);
      abortController.abort();
    });

    // Cancel if client aborts
    req.on('aborted', () => {
      console.log(`Client aborted, aborting request to ${url.pathname}`);
      abortController.abort();
    });

    // Read request body
    body = await new Promise((resolve, reject) => {
      let chunks = [];
      let size = 0;
      const maxSize = 25 * 1024 * 1024; // 25MB limit for audio files

      req.on('data', chunk => {
        size += chunk.length;
        if (size > maxSize) {
          reject(new Error('Request too large'));
          return;
        }
        chunks.push(chunk);
      });

      req.on('end', () => {
        resolve(Buffer.concat(chunks));
      });

      req.on('error', reject);
    });

    // Check if already cancelled before making request
    if (abortController.signal.aborted) {
      res.writeHead(408, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Request cancelled' }));
      return;
    }

    console.log(`Proxying ${req.method} ${url.pathname} to OpenAI (${body.length} bytes)`);

    // Forward request to OpenAI with cancellation support
    const openaiResponse = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Content-Type': contentType,
        'Authorization': `Bearer ${OPENAI_API_KEY}`,
        'User-Agent': 'translator-relay/1.0.0',
      },
      body: body,
      signal: abortController.signal,
    });

    // Check if cancelled during request
    if (abortController.signal.aborted) {
      res.writeHead(408, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Request cancelled' }));
      return;
    }

    console.log(`OpenAI responded with status ${openaiResponse.status} for ${url.pathname}`);

    // Forward response headers (excluding problematic ones)
    const responseHeaders = {};
    for (const [key, value] of openaiResponse.headers) {
      if (!['content-encoding', 'transfer-encoding', 'connection'].includes(key.toLowerCase())) {
        responseHeaders[key] = value;
      }
    }

    res.writeHead(openaiResponse.status, responseHeaders);

    // Stream response back to client with cancellation handling
    if (openaiResponse.body) {
      const reader = openaiResponse.body.getReader();
      
      const pump = async () => {
        try {
          while (true) {
            if (abortController.signal.aborted) {
              break;
            }

            const { done, value } = await reader.read();
            if (done) break;
            
            if (!res.destroyed && res.writable) {
              res.write(value);
            } else {
              break;
            }
          }
        } catch (error) {
          if (!abortController.signal.aborted) {
            console.error('Error streaming response:', error);
          }
        } finally {
          reader.releaseLock();
          if (!res.destroyed) {
            res.end();
          }
        }
      };

      await pump();
    } else {
      res.end();
    }

  } catch (error) {
    // Handle different types of errors
    if (abortController?.signal.aborted || error.name === 'AbortError') {
      console.log(`Request cancelled for ${url.pathname}`);
      if (!res.headersSent) {
        res.writeHead(408, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Request cancelled' }));
      }
    } else {
      console.error(`Error proxying ${url.pathname}:`, error.message);
      if (!res.headersSent) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ 
          error: 'Internal server error',
          message: error.message 
        }));
      }
    }
  }
});

// Graceful shutdown handling
process.on('SIGTERM', () => {
  console.log('Received SIGTERM, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

server.listen(PORT, () => {
  console.log(`🚀 OpenAI Relay server running on port ${PORT}`);
  console.log(`📡 Endpoints:`);
  console.log(`   POST /transcribe - Audio transcription proxy`);
  console.log(`   POST /translate - Chat completion proxy`);
  console.log(`🌏 Ready to serve clients in restricted regions`);
});

export default server; 