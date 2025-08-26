import { createServer } from "node:http";
const PORT = process.env.PORT || 3000;
const server = createServer(async (req, res) => {
  console.log(
    `🔍 Relay received: ${req.method} ${req.url} from ${
      req.headers["user-agent"] || "unknown"
    }`
  );
  // Enable CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization, X-Relay-Secret, X-OpenAI-Key"
  );
  // Handle preflight requests
  if (req.method === "OPTIONS") {
    console.log("✅ Responding to preflight request");
    res.writeHead(200);
    res.end();
    return;
  }
  // Only handle POST to /transcribe
  if (req.method === "POST" && req.url === "/transcribe") {
    console.log("📡 Processing transcribe request...");
    // Validate relay secret
    const relaySecret = req.headers["x-relay-secret"];
    if (!relaySecret) {
      console.log("❌ Missing relay secret");
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized - missing relay secret" }));
      return;
    }
    console.log("🎯 Relay secret validated, sending test response...");
    // Send test response
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({
        message: "came through",
        timestamp: new Date().toISOString(),
        relay: "openai-relay-minimal",
      })
    );
    console.log("✅ Test response sent successfully");
    return;
  }
  // Handle all other requests
  console.log(`❌ Unsupported request: ${req.method} ${req.url}`);
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Endpoint not found" }));
});
server.listen(PORT, () => {
  console.log(`🚀 Minimal OpenAI Relay server running on port ${PORT}`);
  console.log(`📡 Ready to test fallback connections`);
});
