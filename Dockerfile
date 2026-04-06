FROM python:3.11-slim

WORKDIR /app

# Only install curl for health checks (much lighter than build-essential)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install (good for layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

ENV PYTHONPATH="/app:$PYTHONPATH"

# Make entrypoint executable
RUN chmod +x start.sh

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["bash", "start.sh"]