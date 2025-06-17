# OpenAI Relay Service

A robust proxy service for OpenAI API that enables access from regions where OpenAI blocks API requests. Designed specifically for the Translator app with full cancellation support.

## 🎯 Features

- **Regional Access**: Bypass OpenAI regional restrictions by running in allowed regions
- **Dual Endpoints**: Support for both transcription (`/transcribe`) and translation (`/translate`)
- **Robust Cancellation**: Full client-side cancellation support with proper cleanup
- **CORS Support**: Cross-origin requests enabled for web clients
- **Request Size Limits**: 25MB limit for audio files
- **Graceful Shutdown**: Proper signal handling for zero-downtime deployments
- **Error Handling**: Comprehensive error handling with proper HTTP status codes

## 🚀 Local Development

### Prerequisites
- Node.js 18+ 
- OpenAI API key

### Setup
```bash
npm install
export OPENAI_API_KEY=sk-your-key-here
npm run dev
```

The server will start on `http://localhost:3000` with hot reload.

### Test Endpoints
```bash
# Test translation endpoint
curl -X POST http://localhost:3000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Test transcription endpoint (requires audio file)
curl -X POST http://localhost:3000/transcribe \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

## 🌩️ Fly.io Deployment

### Initial Setup
```bash
# Install flyctl CLI
curl -L https://fly.io/install.sh | sh
flyctl auth login

# Launch the app
flyctl launch --name translator-relay --region sin
```

### Set Environment Variables
```bash
flyctl secrets set OPENAI_API_KEY=sk-your-openai-key-here
```

### Deploy
```bash
flyctl deploy
```

Your relay will be available at `https://translator-relay.fly.dev`

### Monitor
```bash
# View logs
flyctl logs

# Check status
flyctl status

# Scale if needed
flyctl scale count 2
```

## 📡 API Endpoints

### POST /transcribe
Proxy to OpenAI's audio transcription API.

**Headers:**
- `Content-Type: multipart/form-data`

**Body:**
- `file`: Audio file (mp3, wav, etc.)
- `model`: Model name (default: whisper-1)
- `language`: Optional language code
- `prompt`: Optional context prompt

### POST /translate  
Proxy to OpenAI's chat completions API.

**Headers:**
- `Content-Type: application/json`

**Body:**
```json
{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "Text to translate"}
  ],
  "temperature": 0.4
}
```

## 🔧 Integration with stage5-api

Update your `stage5-api` Cloudflare Worker to use the relay:

```typescript
// In stage5-api OpenAI client setup
const openai = new OpenAI({
  apiKey: env.OPENAI_API_KEY,
  baseURL: 'https://translator-relay.fly.dev', // Your relay URL
  timeout: 60_000,
  maxRetries: 3,
});

// Update endpoint calls
const transcription = await openai.audio.transcriptions.create({
  // ... same parameters
}); // Will call https://translator-relay.fly.dev/transcribe

const completion = await openai.chat.completions.create({
  // ... same parameters  
}); // Will call https://translator-relay.fly.dev/translate
```

## 🛡️ Security Considerations

### Production Hardening
1. **Restrict CORS**: Update `Access-Control-Allow-Origin` to specific domains
2. **Rate Limiting**: Add rate limiting middleware
3. **Request Validation**: Add stricter input validation
4. **Monitoring**: Set up health checks and monitoring
5. **Secrets Management**: Use Fly.io secrets for sensitive data

### Example Production CORS
```javascript
// Replace in index.js for production
const allowedOrigins = [
  'https://yourdomain.com',
  'https://api.yourdomain.com'
];

const origin = req.headers.origin;
if (allowedOrigins.includes(origin)) {
  res.setHeader('Access-Control-Allow-Origin', origin);
}
```

## 🔍 Troubleshooting

### Common Issues

**Server not starting:**
- Check Node.js version (18+ required)
- Verify `OPENAI_API_KEY` environment variable

**Fly.io deployment fails:**
- Ensure you're logged into flyctl
- Check app name availability
- Verify region supports your plan

**Requests timing out:**
- Check OpenAI API key permissions
- Verify network connectivity from deployment region
- Check Fly.io app logs: `flyctl logs`

**Cancellation not working:**
- Ensure client is properly sending abort signals
- Check network connectivity
- Verify error handling in client code

## 📊 Architecture

```
Client (Electron App) 
    ↓ AbortSignal
stage5-api (Cloudflare Worker)
    ↓ HTTP Request with AbortSignal  
OpenAI Relay (Fly.io - Singapore)
    ↓ Proxied Request with AbortSignal
OpenAI API (Allowed Region)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## 📄 License

ISC License - see LICENSE file for details 