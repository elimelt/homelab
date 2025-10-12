# DevStack

Self-hosted development environment with Traefik reverse proxy, Tailscale VPN, and essential development services.

## Features

- Secure remote access via Tailscale VPN with HTTPS
- Automated setup and configuration
- Full observability stack (Prometheus + Grafana)
- Browser-based development tools (VS Code, JupyterLab)
- PostgreSQL and Redis databases
- Docker management via Portainer

## Services

| Service | Description | URL Path |
|---------|-------------|----------|
| Traefik | Reverse proxy with HTTPS | `/dashboard/` |
| Tailscale | VPN for secure remote access | - |
| Portainer | Docker management UI | `/portainer/` |
| Code Server | VS Code in browser | `/code/` |
| JupyterLab | Interactive notebooks | `/jupyter/` |
| Grafana | Metrics and logs visualization | `/grafana/` |
| Prometheus | Metrics collection | `/prometheus/` |
| Loki | Log aggregation (API only) | - |
| Promtail | Log collector | - |
| PostgreSQL | Relational database | `postgres:5432` |
| Redis | Cache and message broker | `redis:6379` |
| Homepage | Service dashboard | `/` |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Tailscale account
- `htpasswd` utility (apache2-utils package)

### Installation

1. Clone and run setup:
```bash
git clone <your-repo-url> devstack
cd devstack
./setup.sh
```

2. Configure `.env`:
```bash
# Tailscale auth key from https://login.tailscale.com/admin/settings/keys
TAILSCALE_AUTHKEY=tskey-auth-xxxxx
TAILSCALE_DOMAIN=your-machine.tail12345.ts.net

# Single sign-on authentication (htpasswd -nbB admin password | sed -e s/\\$/\\$\\$/g)
BASIC_AUTH_USERS=admin:$$2y$$05$$xxxxx

# Database passwords
POSTGRES_PASSWORD=your-password
REDIS_PASSWORD=your-password
CODE_SERVER_SUDO_PASSWORD=your-password
```

3. Get Tailscale certificates:
```bash
tailscale cert your-machine.tail12345.ts.net
cp your-machine.tail12345.ts.net.* certs/
```

4. Update `traefik-config.yml` with certificate paths:
```yaml
tls:
  certificates:
    - certFile: /certs/your-machine.tail12345.ts.net.crt
      keyFile: /certs/your-machine.tail12345.ts.net.key
```

5. Start services:
```bash
docker-compose up -d
```

Access services at `https://your-machine.tail12345.ts.net`

## Authentication

DevStack uses single sign-on via Traefik's basic auth. All services are configured to bypass their individual authentication and rely on Traefik's auth middleware.

**Single authentication point:**
- Configure `BASIC_AUTH_USERS` in `.env`
- Authenticate once at Traefik level
- Access all services without additional logins

**Services with disabled authentication:**
- Grafana: Anonymous access with Admin role
- Portainer: No admin password required
- Code Server: Password authentication disabled
- JupyterLab: Token authentication disabled

## Monitoring & Logging

### Metrics (Prometheus + Grafana)

Import Traefik dashboard in Grafana:
- Dashboard ID: 11462 or 4475
- Data source: Prometheus

Metrics collected:
- Container metrics (CPU, memory, network)
- System metrics (node-exporter)
- Traefik metrics (requests, latency, status codes)

Prometheus queries:
```promql
# Request rate by service
rate(traefik_service_requests_total[5m])

# 95th percentile latency
histogram_quantile(0.95, rate(traefik_service_request_duration_seconds_bucket[5m]))

# HTTP status codes
sum by (code) (rate(traefik_entrypoint_requests_total[5m]))
```

### Logs (Loki + Grafana)

Loki aggregates logs from all services and Traefik access logs. Access via Grafana Explore or create dashboards.

LogQL queries:
```logql
# All Traefik access logs
{job="traefik"}

# Filter by service
{job="traefik", router_name=~"grafana.*"}

# Filter by status code
{job="traefik"} | json | status >= 400

# Filter by IP
{job="traefik"} | json | client_addr =~ "100.64.*"

# Filter by user
{job="traefik"} | json | client_username = "admin"

# All container logs
{job="docker"}

# Specific container
{job="docker", container="postgres"}

# Search in logs
{job="docker"} |= "error"
```

Access logs in Grafana:
1. **Pre-built Dashboard**: Go to Grafana → Dashboards → "DevStack Logs"
2. **Explore**: Go to Grafana → Explore → Select "Loki" data source
3. Use LogQL queries above to filter and search logs

Pre-configured dashboards:
- **DevStack Logs**: Real-time access logs, request rates, error tracking
- **DevStack Overview**: System metrics and service health
- **Traefik**: Official Traefik dashboard with detailed proxy metrics
- **Redis**: Redis performance and usage metrics
- **Cost Usage**: Resource usage tracking

All dashboards are version-controlled in `grafana/dashboards/` and automatically provisioned on startup.

## Management

### Common Commands

```bash
docker-compose logs -f [service]      # View logs
docker-compose restart [service]      # Restart service
docker-compose pull && docker-compose up -d  # Update services
docker-compose down                   # Stop all
docker-compose down -v                # Stop and remove data
```

### Access Logs

View all incoming connections and IPs:

```bash
# Real-time access log
tail -f logs/access.log

# View with jq for formatted output
tail -f logs/access.log | jq

# Filter by IP
grep "ClientAddr" logs/access.log | jq '.ClientAddr'

# Filter by service
grep "RouterName" logs/access.log | jq 'select(.RouterName | contains("grafana"))'

# View authentication attempts
grep "ClientUsername" logs/access.log | jq '{time: .time, ip: .ClientAddr, user: .ClientUsername, status: .DownstreamStatus}'
```

Log fields include:
- `ClientAddr` - Source IP and port
- `ClientUsername` - Authenticated username
- `RequestMethod` - HTTP method
- `RequestPath` - URL path
- `RouterName` - Which service was accessed
- `DownstreamStatus` - HTTP status code
- `Duration` - Request duration in nanoseconds

### Database Access

```bash
# PostgreSQL
docker exec -it postgres psql -U devuser -d devdb

# Redis
docker exec -it redis redis-cli -a your-password
```

## Data Persistence

Docker volumes:
- `portainer_data` - Portainer configuration
- `postgres_data` - PostgreSQL databases
- `redis_data` - Redis data
- `grafana_data` - Grafana dashboards
- `prometheus_data` - Metrics history
- `loki_data` - Log storage

### Backup

**Grafana dashboards and datasources:**
```bash
./export-grafana.sh
```

This creates:
- `grafana-export/datasources/` - All datasource configurations
- `grafana-export/dashboards/` - All dashboard JSON files
- `grafana-backup-TIMESTAMP.tar.gz` - Compressed archive

**Database volumes:**
```bash
docker run --rm -v devstack_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

**Restore Grafana:**
```bash
cp grafana-export/datasources/* grafana/datasources/
cp grafana-export/dashboards/* grafana/dashboards/
docker-compose restart grafana
```

## Security

- Change all default passwords in `.env`
- Use ephemeral Tailscale auth keys with expiration
- Configure Tailscale ACLs to restrict access
- Keep services updated regularly
- Enable Tailscale MFA
- Review Traefik access logs periodically

## Troubleshooting

### Services won't start
```bash
docker-compose logs
cat .env | grep -v "^#" | grep -v "^$"
sudo netstat -tulpn | grep -E ':(80|443|5432|6379)'
```

### Can't access services
```bash
# Verify Tailscale connection
tailscale status

# Check Traefik dashboard at https://your-domain/dashboard/

# Test certificate
openssl s_client -connect your-domain:443 -servername your-domain
```

### Homepage multiplexor
The homepage provides a multiplexor interface for managing multiple services. Services that support iframe embedding (Code Server, JupyterLab, Prometheus, Traefik) can be added to the grid. Services with CSP restrictions (Grafana, Portainer) are marked with ↗ and open in new tabs when clicked.

### Certificate errors
```bash
tailscale cert --force your-machine.tail12345.ts.net
cp your-machine.tail12345.ts.net.* certs/
docker-compose restart traefik
```

### Prometheus not collecting metrics
```bash
curl http://localhost:9090/api/v1/targets
docker exec traefik wget -qO- http://localhost:8080/metrics | head
```

## Customization

### Adding Services

Add to `docker-compose.yml` with Traefik labels:
```yaml
labels:
  - "traefik.enable=true"
  - "traefik.http.routers.myservice.rule=Host(`${DOMAIN}`) && PathPrefix(`/myservice`)"
  - "traefik.http.routers.myservice.entrypoints=websecure"
  - "traefik.http.routers.myservice.tls=true"
```

### Workspace Paths

Edit `.env`:
```bash
CODE_SERVER_WORKSPACE_PATH=~/my-projects
JUPYTER_WORKSPACE_PATH=./my-notebooks
```
