### Chat API Client Guide (WebSocket + History + Events)

This guide shows how to build a small chat UI using the real-time WebSocket API and the time‑paginated HTTP history APIs. Focus: infinite scrolling and optional presence enhancements.

## Endpoints
- WebSocket (real‑time chat per channel): `/ws/chat/{channel}`
- Chat history (newest‑first, time‑paginated): `GET /chat/{channel}/history?before=ISO8601&limit=50`
- Generic events feed (optional, for presence/analytics): `GET /events?topic=...&type=...&before=ISO8601&limit=100`

Typical public URLs
- Local compose: `ws://localhost:10000/ws/chat/general` and `http://localhost:10000/chat/general/history`
- Via reverse proxy/HTTPS: `wss://YOUR_DOMAIN/ws/chat/general` and `https://YOUR_DOMAIN/chat/general/history`

## Event Shapes (over WebSocket)
- Incoming chat message (server → client):
  ```json
  { "type": "chat_message", "channel": "general", "sender": "203.0.113.10:12345", "text": "hello", "timestamp": "2025-11-28T12:34:56.789Z" }
  ```
- Ping (keepalive; ignore in UI):
  ```json
  { "type": "ping" }
  ```
- Outgoing (client → server) to send a message:
  ```json
  { "text": "hello world" }
  ```

## Infinite Scrolling (History API)
- Use `GET /chat/{channel}/history?before={ISO}&limit=50` to load messages in pages.
- Response (newest‑first):
  ```json
  { "messages": [ { "type":"chat_message", "channel":"general", "sender":"...", "text":"...", "timestamp":"..." }, ... ],
    "next_before": "2025-11-28T12:00:00Z" }
  ```
- Client flow:
  1) On page load, call history with `before=now` (or omit).
  2) Render ascending for display (oldest at top): reverse the returned list.
  3) On scroll‑top, call history again with `before=next_before` to load older pages. Prepend to the list (keep scroll position stable).
  4) Stop when an empty page is returned.

Minimal JS for history pagination
```js
let nextBefore = null; // ISO string
const channel = "general";
const pageSize = 50;

async function fetchHistory(initial = false) {
  const params = new URLSearchParams();
  if (nextBefore && !initial) params.set("before", nextBefore);
  params.set("limit", String(pageSize));
  const res = await fetch(`/chat/${encodeURIComponent(channel)}/history?` + params, { credentials: "include" });
  const body = await res.json();
  const msgs = body.messages || [];
  // display ascending
  msgs.reverse().forEach(renderMessageAtTop);
  nextBefore = body.next_before || nextBefore;
}
```

## Real‑Time (WebSocket)
- One connection per open channel view: `/ws/chat/{channel}`
- On `message`:
  - Ignore `{type:"ping"}`
  - For `{type:"chat_message"}`, append to the bottom of the list
- Reconnect with exponential backoff if closed

Minimal JS for WebSocket
```js
const wsURL = (location.protocol === "https:" ? "wss://" : "ws://") +
  location.host + "/ws/chat/" + encodeURIComponent("general");
let ws;

function openWS() {
  ws = new WebSocket(wsURL);
  ws.onmessage = (e) => {
    try {
      const evt = JSON.parse(e.data);
      if (evt.type === "ping") return;
      if (evt.type === "chat_message") renderMessageAtBottom(evt);
    } catch { /* ignore */ }
  };
  ws.onclose = () => setTimeout(openWS, 1000); // naive backoff for brevity
}

function sendMessage(text) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ text }));
  }
}

openWS();
```

## Extra Events for Enhanced Clients (Presence, Analytics)
Your server also emits visit presence events (decoupled from chat) that you can use to enhance the UI:
- WebSocket stream: `/ws/visitors` forwards Redis pubsub messages and pings
- Persisted events: `/events?topic=visitor_updates&type=join|leave&before=...&limit=...`
- Shapes:
  - Join: `{ "type":"join", "visitor": { "ip": "...", "location": {...}, "connected_at": "..." } }`
  - Leave: `{ "type":"leave", "ip": "..." }`

Ideas:
- Show “online now” and recent arrivals using `/ws/visitors` in real‑time
- Render presence history (e.g., activity sparkline) using `/events` with `topic=visitor_updates`

## State Model (Client)
- `connection`: "connecting" | "open" | "closed" | "error"
- `channel`: string (e.g., "general")
- `messages`: array of chat_message events (display ascending)
- `nextBefore`: ISO string for fetching older pages
- Optional:
  - `unsentQueue`: pending texts to flush on reconnect
  - `presence`: latest presence snapshot and/or last N join/leave events

## Tips
- Dedup on reconnect: for history + live overlap, dedup by `(timestamp, sender, text)`
- Prefer WebSockets over polling; use history only for initial page and infinite scroll
- For multi‑channel UIs, one WS per visible channel tab/pane
- CORS/Origin: ensure your page origin is allowed by the server’s CORS config


