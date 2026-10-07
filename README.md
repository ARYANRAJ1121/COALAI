# COALAI

**Cost-Optimized AI Infrastructure**

Production-grade AI execution infrastructure for reliable, cost-aware LLM and agent workloads.

## Quickstart (local dev)

```bash
# Start infrastructure
docker compose up -d postgres redis ollama

# Pull a model into Ollama
docker exec coalai_ollama ollama pull llama3.2:3b

# Install Python package
pip install -e ".[dev]"

# Apply database migrations
alembic upgrade head

# Create a tenant and get an API key
python scripts/provision.py create --name "my-app"

# Start the gateway
uvicorn coalai.main:app --reload --port 8000
```

## Run tests

```bash
# Unit tests (no infrastructure needed)
pytest tests/unit/ -v

# Integration tests (requires Postgres + Redis)
pytest tests/integration/ -v -m integration
```
