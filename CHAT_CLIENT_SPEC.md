### Chat Client Spec (Barebones HTML/CSS/JS)

This document describes a minimal, event-driven chat client for humans using plain HTML/CSS/JS. The server is Redis-backed and communicates over WebSockets.

## Overview
- Connect to a channel-specific WebSocket: `/ws/chat/{channel}`
- Send messages by posting JSON with a `text` field over the WebSocket
- Receive events as JSON strings
- No auth or history for now; focus is real-time broadcast within a channel

## Endpoints
- Local (docker-compose): `ws://localhost:10000/ws/chat/{channel}`
- Behind HTTPS/Traefik: `wss://{your-domain}/ws/chat/{channel}`
- Path param:
  - `channel` (string): selects the chat room (e.g., `general`)

## Event Shapes
- Incoming from server:
  - Chat message:
    ```
    {
      "type": "chat_message",
      "channel": "general",
      "sender": "203.0.113.10:12345678",
      "text": "hello",
      "timestamp": "2025-11-28T12:34:56.789Z"
    }
    ```
  - Ping (keepalive):
    ```
    { "type": "ping" }
    ```
    - Client should ignore this (no response required).

- Outgoing from client:
  - Send message:
    ```
    { "text": "hello world" }
    ```
    - Server will enrich with `type`, `channel`, `sender`, and `timestamp`.

## Minimal State Model (Client)
- `connection`: "connecting" | "open" | "closed" | "error"
- `channel`: string (e.g., "general")
- `messages`: array of `{ sender, text, timestamp }`
- `inputText`: string (message compose box)
- Optional:
  - `unsentQueue`: array of pending messages for retry on reconnect
  - `reconnectAttempts`: number (exponential backoff)

## Reconnection Strategy (Recommended)
- On `close`/`error`, wait with exponential backoff and reconnect (e.g., 500ms, 1s, 2s, max 10s)
- Clear/flush `unsentQueue` when connection re-opens
- Ignore `ping` messages; they indicate the server is alive

## Minimal HTML/CSS/JS Example
This is intentionally barebones. It connects to the `general` channel and allows sending text messages.

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Barebones Chat</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 0; background: #111; color: #eee; }
    .container { max-width: 720px; margin: 0 auto; padding: 16px; }
    .status { font-size: 12px; color: #9aa; margin-bottom: 8px; }
    .messages { height: 60vh; overflow-y: auto; padding: 8px; background: #181818; border: 1px solid #333; border-radius: 6px; }
    .msg { padding: 6px 8px; margin: 6px 0; background: #202020; border-radius: 4px; }
    .meta { font-size: 11px; color: #9aa; margin-bottom: 2px; }
    form { display: flex; gap: 8px; margin-top: 10px; }
    input[type="text"] { flex: 1; padding: 10px; border-radius: 4px; border: 1px solid #333; background: #151515; color: #eee; }
    button { padding: 10px 12px; border: 0; border-radius: 4px; background: #2a7; color: #111; cursor: pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="container">
    <div class="status" id="status">connecting…</div>
    <div class="messages" id="messages"></div>
    <form id="chat-form">
      <input id="chat-input" type="text" placeholder="Say something…" autocomplete="off" />
      <button id="send-btn" type="submit">Send</button>
    </form>
  </div>

  <script>
    (function () {
      const channel = "general"; // change to your channel
      const wsUrl = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/chat/" + encodeURIComponent(channel);
      const localFallback = "ws://localhost:10000/ws/chat/" + encodeURIComponent(channel);

      const statusEl = document.getElementById("status");
      const msgsEl = document.getElementById("messages");
      const formEl = document.getElementById("chat-form");
      const inputEl = document.getElementById("chat-input");
      const sendBtn = document.getElementById("send-btn");

      let ws = null;
      let reconnectDelay = 500; // ms
      const maxDelay = 10000;

      const state = {
        connection: "closed",
        channel,
        messages: [],
        unsentQueue: []
      };

      function setConnection(status) {
        state.connection = status;
        statusEl.textContent = status;
        sendBtn.disabled = status !== "open";
      }

      function renderMessage(msg) {
        const item = document.createElement("div");
        item.className = "msg";
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = `${msg.sender ?? "unknown"} • ${new Date(msg.timestamp ?? Date.now()).toLocaleTimeString()}`;
        const text = document.createElement("div");
        text.textContent = msg.text;
        item.appendChild(meta);
        item.appendChild(text);
        msgsEl.appendChild(item);
        msgsEl.scrollTop = msgsEl.scrollHeight;
      }

      function connect() {
        // Try current host first, then local fallback if that fails immediately
        tryOpen(wsUrl, (opened) => {
          if (!opened) tryOpen(localFallback);
        });
      }

      function tryOpen(url, cb) {
        setConnection("connecting");
        ws = new WebSocket(url);

        ws.onopen = () => {
          setConnection("open");
          // Flush any unsent messages
          while (state.unsentQueue.length > 0) {
            ws.send(JSON.stringify({ text: state.unsentQueue.shift() }));
          }
          reconnectDelay = 500; // reset backoff
          cb && cb(true);
        };

        ws.onmessage = (evt) => {
          let data = null;
          try { data = JSON.parse(evt.data); } catch { /* ignore non-JSON */ return; }
          if (data.type === "ping") return; // keepalive
          if (data.type === "chat_message" && data.channel === state.channel) {
            state.messages.push({ sender: data.sender, text: data.text, timestamp: data.timestamp });
            renderMessage(data);
          }
        };

        ws.onerror = () => {
          cb && cb(false);
        };

        ws.onclose = () => {
          setConnection("closed");
          setTimeout(connect, reconnectDelay);
          reconnectDelay = Math.min(reconnectDelay * 2, maxDelay);
        };
      }

      formEl.addEventListener("submit", (e) => {
        e.preventDefault();
        const text = (inputEl.value || "").trim();
        if (!text) return;
        if (state.connection === "open" && ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ text }));
        } else {
          state.unsentQueue.push(text);
        }
        inputEl.value = "";
      });

      connect();
    })();
  </script>
</body>
</html>
```

## Notes & Best Practices
- One WebSocket per open channel view is sufficient; avoid creating multiple per channel tab
- Use incremental rendering for efficiency; append messages rather than re-rendering the whole list
- Don’t block on pings; they’re informational keepalives
- For production, consider:
  - Auth and user identity (sender name/avatar)
  - Channel permissions and moderation
  - Message persistence with Redis lists (e.g., `lpush` per channel) and a REST endpoint to fetch recent history
  - Backoff jitter and a max retry budget


