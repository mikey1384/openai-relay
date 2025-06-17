import { createServer } from 'node:http';
import { fetch } from 'undici';

const PORT = process.env.PORT || 3000;

// Enhanced relay for translator app - supports both transcription and translation
const server = createServer(async (req, res) => {
  console.log(`🔍 Incoming ${req.method} ${req.url} from ${req.headers['user-agent'] || 'unknown'}`);

  // Enable CORS for all origins (restrict in production)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Relay-Secret, X-OpenAI-Key');

  // Handle preflight requests
  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  const url = new URL(req.url, 'http://localhost');

  // Handle async job processing endpoint
  if (url.pathname === '/queue' && req.method === 'POST') {
    await handleQueueJob(req, res);
    return;
  }

  if (req.method !== 'POST') {
    res.writeHead(405, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Method not allowed' }));
    return;
  }

  let targetUrl;
  let contentType = req.headers['content-type'] || 'application/json';

  // Route to appropriate OpenAI endpoint - support both direct and SDK paths
  if (url.pathname === '/transcribe' || url.pathname === '/v1/audio/transcriptions') {
    targetUrl = 'https://api.openai.com/v1/audio/transcriptions';
  } else if (url.pathname === '/translate' || url.pathname === '/v1/chat/completions') {
    targetUrl = 'https://api.openai.com/v1/chat/completions';
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Endpoint not found' }));
    return;
  }

  // Validate relay authorization
  const relaySecret = req.headers['x-relay-secret'];
  if (!relaySecret) {
    res.writeHead(401, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Unauthorized - missing relay secret' }));
    return;
  }

  // Extract OpenAI API key from request headers
  const openaiApiKey = req.headers['x-openai-key'];
  if (!openaiApiKey) {
    res.writeHead(400, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Missing OpenAI API key' }));
    return;
  }

  try {
    // Set up abort controller for cancellation
    const abortController = new AbortController();
    
    // Handle client disconnect
    req.on('close', () => {
      console.log('Client disconnected, aborting request');
      abortController.abort();
    });

    // Set up timeout (5 minutes for transcription, 2 minutes for translation)
    const timeoutMs = url.pathname.includes('transcribe') ? 5 * 60 * 1000 : 2 * 60 * 1000;
    const timeoutId = setTimeout(() => {
      console.log(`Request timeout after ${timeoutMs}ms`);
      abortController.abort();
    }, timeoutMs);

    // Stream request body
    const chunks = [];
    for await (const chunk of req) {
      if (abortController.signal.aborted) break;
      chunks.push(chunk);
    }

    if (abortController.signal.aborted) {
      clearTimeout(timeoutId);
      if (!res.headersSent) {
        res.writeHead(408, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Request cancelled' }));
      }
      return;
    }

    const body = Buffer.concat(chunks);

    // Forward to OpenAI
    console.log(`Forwarding to ${targetUrl}...`);
    const response = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openaiApiKey}`,
        'Content-Type': contentType,
        'User-Agent': 'translator-relay/1.0.0',
      },
      body,
      signal: abortController.signal,
    });

    clearTimeout(timeoutId);

    // Forward response status and headers
    res.writeHead(response.status, {
      'Content-Type': response.headers.get('content-type') || 'application/json',
      'Access-Control-Allow-Origin': '*',
    });

    // Stream response back
    if (response.body) {
      const reader = response.body.getReader();
      const pump = async () => {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          if (!res.write(value)) {
            await new Promise(resolve => res.once('drain', resolve));
          }
        }
        res.end();
      };
      await pump();
    } else {
      res.end();
    }

  } catch (error) {
    // Handle different types of errors
    if (error.name === 'AbortError') {
      console.log(`Request cancelled for ${url.pathname}`);
      if (!res.headersSent) {
        res.writeHead(408, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Request cancelled' }));
      }
    } else {
      console.error(`Error processing ${url.pathname}:`, error);
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

// Handle async job processing - gets everything from Worker
async function handleQueueJob(req, res) {
  console.log(`🔥 Queue request - Method: ${req.method}, URL: ${req.url}`);

  const relaySecret = req.headers['x-relay-secret'];
  if (!relaySecret) {
    console.log(`❌ Missing relay secret`);
    res.writeHead(401, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Unauthorized - missing relay secret' }));
    return;
  }

  try {
    console.log(`📖 Reading request body...`);
    // Read the job data from Worker
    let body = '';
    let totalSize = 0;

    for await (const chunk of req) {
      totalSize += chunk.length;
      if (totalSize > 10 * 1024 * 1024) { // 10MB limit for job data
        throw new Error(`Request body too large: ${totalSize} bytes`);
      }
      body += chunk;
    }

    console.log(`✅ Body read complete: ${totalSize} bytes`);
    const { jobId, job, callbackUrl, audioUrl, openaiApiKey, relaySecret: workerSecret } = JSON.parse(body);

    console.log(`🎯 Processing job: ${jobId}`);
    console.log(`📂 Job details - file: ${job.fileName}, size: ${job.fileSize}, model: ${job.model}`);

    if (!openaiApiKey || !audioUrl || !callbackUrl || !workerSecret) {
      console.log(`❌ Missing required parameters`);
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Missing required parameters' }));
      return;
    }

    console.log(`🎯 Processing async job: ${jobId}`);

    // Process the job in the background (don't wait)
    processJobAsync(jobId, job, callbackUrl, audioUrl, openaiApiKey, workerSecret).catch(err =>
      console.error(`Failed to process job ${jobId}:`, err)
    );

    // Respond immediately to avoid Cloudflare timeout
    console.log(`✅ Sending success response for job ${jobId}`);
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ success: true, jobId }));

  } catch (error) {
    console.error('💥 Error handling queue job:', error);
    res.writeHead(500, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Failed to process job' }));
  }
}

// Process job asynchronously - gets everything from Worker
async function processJobAsync(jobId, job, callbackUrl, audioUrl, openaiApiKey, relaySecret) {
  try {
    console.log(`📋 Starting async processing for job ${jobId}...`);

    // Get audio file from the URL provided by Worker
    console.log(`🎵 Fetching audio from: ${audioUrl}`);
    const audioResponse = await fetch(audioUrl, {
      headers: {
        'X-Relay-Secret': relaySecret,
      },
    });

    if (!audioResponse.ok) {
      throw new Error(`Failed to fetch audio: ${audioResponse.status} ${audioResponse.statusText}`);
    }

    console.log(`✅ Audio fetch successful: ${audioResponse.status}`);
    const audioData = await audioResponse.arrayBuffer();
    console.log(`📦 Audio data loaded: ${audioData.byteLength} bytes`);

    // Create form data for OpenAI
    console.log(`🤖 Creating OpenAI request...`);
    const formData = new FormData();
    formData.append('file', new Blob([audioData], { type: 'audio/mpeg' }), job.fileName);
    formData.append('model', job.model);
    formData.append('response_format', 'verbose_json');
    formData.append('timestamp_granularities', 'word');
    formData.append('timestamp_granularities', 'segment');
    
    if (job.language) formData.append('language', job.language);
    if (job.prompt) formData.append('prompt', job.prompt);

    console.log(`🚀 Sending to OpenAI API...`);
    // Call OpenAI API with the key provided by Worker
    const openaiResponse = await fetch('https://api.openai.com/v1/audio/transcriptions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${openaiApiKey}`,
        'User-Agent': 'translator-relay/1.0.0',
      },
      body: formData,
    });

    console.log(`🤖 OpenAI responded: ${openaiResponse.status}`);

    if (!openaiResponse.ok) {
      const errorText = await openaiResponse.text();
      throw new Error(`OpenAI API error: ${openaiResponse.status} ${errorText}`);
    }

    const transcript = await openaiResponse.json();
    console.log(`✅ Job ${jobId} completed successfully - transcript length: ${transcript.text?.length || 0} chars`);

    // Send result back to Worker via callback
    console.log(`📞 Sending result to callback: ${callbackUrl}`);
    const callbackResponse = await fetch(callbackUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Relay-Secret': relaySecret,
      },
      body: JSON.stringify({ transcript }),
    });

    if (!callbackResponse.ok) {
      console.error(`❌ Callback failed: ${callbackResponse.status} ${callbackResponse.statusText}`);
    } else {
      console.log(`✅ Callback successful: ${callbackResponse.status}`);
    }

  } catch (error) {
    console.error(`❌ Job ${jobId} failed:`, error);

    // Send error back to Worker via callback
    try {
      console.log(`📞 Sending error to callback: ${callbackUrl}`);
      const callbackResponse = await fetch(callbackUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Relay-Secret': relaySecret,
        },
        body: JSON.stringify({ error: error.message }),
      });

      if (!callbackResponse.ok) {
        console.error(`❌ Error callback failed: ${callbackResponse.status} ${callbackResponse.statusText}`);
      } else {
        console.log(`✅ Error callback successful: ${callbackResponse.status}`);
      }
    } catch (callbackError) {
      console.error(`💥 Failed to send error callback for job ${jobId}:`, callbackError);
    }
  }
}

// Graceful shutdown handling
process.on('SIGTERM', () => {
  console.log('Received SIGTERM, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  console.log('Received SIGINT, shutting down gracefully');
  server.close(() => {
    console.log('Server closed');
    process.exit(0);
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Relay server listening on port ${PORT}`);
  console.log(`📡 Ready to proxy requests to OpenAI API`);
});
