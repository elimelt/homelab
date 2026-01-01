FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget docker.io && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN wget -O GeoLite2-City.mmdb "https://github.com/P3TERX/GeoLite.mmdb/raw/download/GeoLite2-City.mmdb" || \
    echo "GeoIP database not available, will use fallback"

# Copy application code
COPY main.py .
RUN mkdir -p /app/api
COPY __init__.py bus.py events.py state.py db.py middleware.py redis_debug.py /app/api/
COPY controllers /app/api/controllers
COPY producers /app/api/producers
COPY agents /app/api/agents
COPY batch /app/api/batch

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--no-access-log", "--log-level", "info"]

