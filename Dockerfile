FROM python:3.12-slim

# Prevent .pyc files, force unbuffered stdout (important for structured logging)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies needed by asyncpg
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source
# (layer caching: deps change less often than source)
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Copy source and migrations
COPY src/ src/
COPY migrations/ migrations/
COPY alembic.ini .

# Create non-root user
RUN useradd -m -u 1000 coalai && chown -R coalai:coalai /app
USER coalai

EXPOSE 8000

# Run migrations then start the server
CMD ["sh", "-c", "\
    alembic upgrade head && \
    uvicorn coalai.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 2 \
        --log-config /dev/null \
"]
