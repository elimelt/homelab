# Homelab

## Quick Start

### Prerequisites

- Docker + docker-compose
- Tailscale account
- `/dev/net/tun` available on host

### Setup

```bash
git clone <repo-url> homelab && cd homelab
./setup.sh
```

Edit `.env`:
```bash
TAILSCALE_AUTHKEY=tskey-auth-xxxxx          # from https://login.tailscale.com/admin/settings/keys
TAILSCALE_DOMAIN=machine.tail12345.ts.net   # your Tailscale FQDN
BASIC_AUTH_USERS=admin:$$2y$$05$$xxxxx      # htpasswd -nbB admin pass | sed 's/\$/\$\$/g'
POSTGRES_PASSWORD=changeme
AUGMENT_API_TOKEN=xxx                        # optional: enables AI agents
GEMINI_API_KEY=xxx                           # optional: enables Gemini agent
```

Get TLS certs from Tailscale and start:
```bash
tailscale cert machine.tail12345.ts.net
cp machine.tail12345.ts.net.* certs/
docker-compose up -d
```

Access at `https://<TAILSCALE_DOMAIN>/`

### Common Commands

```bash
docker-compose logs -f [service]
docker-compose restart [service]
docker-compose down
docker exec -it postgres psql -U devuser -d devdb
docker exec -it redis redis-cli
```

---

## Architecture

### Data Flow

(😲 LLMs are actually good at this now)

```
        Internet
   ┌──────────────────────────────────────────────────────────────────┐
   │                                                                  │
   └─────▶ Tailscale Funnel (:not 443)                                │
                 │                                                    │
                 │                                                    │
                 └──▶ Public API ──────▶ Redis (pub/sub, cache)       │
                      (on elimelt.com)   Postgres (DB)                │ 
                                                ▲                     │
                                                │                     │
                         shared netork ns       │                     │
                                                │                     │
                                                │                     │
                      Internal API ─────────────┘                     │
                      (not on elimelt.com)                            │
                                                                      │
                                                                      │
Tailscale VPN ──────▶ Traefik (:443) ──┬──▶ Homepage (nginx)          │
                      (reverse proxy)  ├──▶ Grafana                   │
                                       ├──▶ Prometheus                │
                                       ├──▶ Loki                      │
                                       └──▶ Internal API              │
                                                                      │
       ┌─────────── shares network namespace ─────────────────────────┘
       │
       ▼
Internal API
       │
       ├──▶ Agents ──▶ chat channels via Redis pub/sub
       │       │
       │       ▼
       ├──▶ Python Sandbox
       └──▶ Notes Sync (GitHub → Postgres)

Observability:
  Promtail ──▶ Loki (logs)
  Node-exporter + cAdvisor ──▶ Prometheus (metrics) ──▶ Grafana
```

### Service Map

| Service | Port | Purpose |
|---------|------|---------|
| traefik | 80, 443 | Reverse proxy, TLS termination |
| tailscale | - | VPN access (shares traefik network) |
| homepage | - | Dashboard UI |
| public-api | 443 (via Tailscale Funnel) | External API for elimelt.com |
| internal-api | - | Agents, admin, notes sync |
| redis | 6379 | Pub/sub, caching |
| postgres | 5432 | Persistent storage |
| grafana | 3000 | Dashboards |
| prometheus | 9090 | Metrics |
| loki | 3100 | Logs |
| promtail | - | Log shipper |
| node-exporter | 9100 | Host metrics |
| cadvisor | 8080 | Container metrics |

---

## Component Breakdown

### Traefik

Reverse proxy handling all ingress traffic.

- **Routing**: Label-based Docker discovery. Dual-host matching for public domain + Tailscale domain.
- **TLS**: Loads certs from `./certs/`. Tailscale cert as default.
- **Auth**: BasicAuth middleware on most routes.
- **Metrics**: Prometheus endpoint with router/service labels.
- **Logs**: JSON access logs to `./logs/`.

Key paths: `/dashboard` (Traefik UI), `/grafana`, `/prometheus`, `/loki`, `/` (homepage).

### Tailscale

VPN sidecar for secure remote access.

- **Network**: `network_mode: service:traefik` — shares Traefik's network namespace.
- **Capabilities**: `NET_ADMIN`, `SYS_MODULE`, kernel tun device.
- **State**: Persisted to `./tailscale/state/`.

Environment: `TS_AUTHKEY`, `TS_HOSTNAME` (default: `devstack`).

### Redis

In-memory data store for real-time features.

- **Persistence**: AOF (appendonly, fsync every second) + RDB snapshots.
- **Memory**: 256MB default, LRU eviction.
- **Health**: Docker healthcheck via `redis-cli ping`.

Usage patterns:
- Pub/sub channels: `visitor_updates`, `chat:{channel}`
- Visitor state with TTL
- Visit log (capped list)
- Agent rate limiting and deduplication

### PostgreSQL

Persistent storage with vector search support.

- **Image**: `pgvector/pgvector:pg16` (Postgres 16 + pgvector extension).
- **Pool**: 2-10 connections via `psycopg_pool.AsyncConnectionPool`.
- **Migrations**: Version-tracked SQL in `api/db/migrations/`.

Tables:
| Table | Purpose |
|-------|---------|
| `chat_messages` | Chat history with soft delete, threading |
| `events` | Generic event log (JSONB payload) |
| `visitor_stats` | Per-IP analytics with geo |
| `notes_documents` | Markdown notes with `tsvector` + `vector(384)` |
| `notes_categories`, `notes_tags` | Note organization |
| `notes_sync_jobs` | GitHub sync tracking |
| `w2m_events`, `w2m_availabilities` | When2Meet clone |

### Public API

FastAPI backend for `elimelt.com`. Publicly exposed via Tailscale Funnel. See /docs for endpoints

Environment: `REDIS_*`, `POSTGRES_*`, `CORS_ORIGINS`, `NOTES_SYNC_SECRET`, `GITHUB_TOKEN`.

### Internal API

FastAPI backend for agents and admin. Not publicly exposed.

**Agents**:

Autonomous AI agents that participate in chat channels.

| Agent | Model | SDK |
|-------|-------|-----|
| Augment | `sonnet4.5` (configurable) | `auggie_sdk` |
| Gemini | `gemini-2.5-flash` | `google-genai` |

Tools available to agents:
- `search_notes`, `get_note` — query notes database
- `run_python` — execute code in sandbox
- `web-search`, `web-fetch` — web access
- `query_chat` — read chat history

**Sandbox**:

Isolated Python execution for agent code.

- `--network none`, `--read-only`, `--cap-drop ALL`
- 128MB memory, 50% CPU, 64 PIDs max
- 30s timeout, 10 executions/minute rate limit
- Result caching (5 min TTL)

Packages: numpy, pandas, scipy, sympy, matplotlib, scikit-learn, requests, beautifulsoup4.

**Background jobs**:
- Notes sync: GitHub → Postgres every 6 hours

Environment: `AUGMENT_API_TOKEN`, `GEMINI_API_KEY`, `ENABLE_*_AGENT`, `SANDBOX_*`.

### Homepage

Dashboard UI served by nginx.

- **Tech**: Vanilla ES6 JavaScript, CSS Grid.
- **Features**: Tiled iframe grid, AI chat panel, health monitoring.
- **State**: Persisted to localStorage.

Manager classes:
- `FrameManager` — iframe lifecycle
- `ChatManager` — AI chat via SSE streaming
- `HealthManager` — API polling
- `KeyboardManager` — shortcuts (`ctrl+n/p` nav, `ctrl+w` close, `ctrl+/` chat)

Chat uses SSE streaming to `/api/augment/chat`, parses Augment XML response format.

### Observability Stack

**Prometheus**: Scrapes metrics from traefik, public-api, internal-api, node-exporter, cadvisor every 15s.

**Node-exporter**: Host metrics (CPU, memory, disk, network).

**cAdvisor**: Container metrics. Docker-only mode, reduced metric set.

**Loki**: Log aggregation. 7-day retention, filesystem storage.

**Promtail**: Ships logs to Loki. Scrapes:
- Traefik JSON access logs (parses status, duration, router)
- Docker container logs via socket discovery

**Grafana**: Dashboards at `/grafana`. Anonymous admin access (auth handled by Traefik).

Dashboards: `SystemOverview.json`, `ApiPerformance.json`, `ApplicationHealth.json`.

