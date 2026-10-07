<div align="center">

# COALAI

**Cost-Optimized AI Infrastructure**

*Production-grade AI execution infrastructure for reliable, cost-aware LLM and agent workloads.*

[![CI](https://github.com/ARYANRAJ1121/COALAI/actions/workflows/ci.yml/badge.svg)](https://github.com/ARYANRAJ1121/COALAI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![License](https://img.shields.io/badge/license-MIT-blue)

</div>

---

## What is COALAI?

COALAI is an **AI gateway and infrastructure platform** that sits between your application and LLM providers. It handles everything that every AI-powered application needs but shouldn't have to build itself:

- **Auth** — API key management with tenant isolation
- **Rate limiting** — per-tenant RPM/RPD enforcement
- **Durable accounting** — every request is durably recorded before a response is returned
- **Reliability** — retry with exponential backoff, jitter, and Retry-After support
- **Provider abstraction** — swap or fallback between LLM providers without changing your app
- **OpenAI-compatible API** — drop-in replacement; no SDK changes needed

> COALAI runs **fully local** with Ollama — no paid API keys required for development or demos.

---

## Architecture

```
Client
  │  Bearer <api_key>
  ▼
┌─────────────────────────────────────────────┐
│              COALAI Gateway                  │
│                                             │
│  Auth → Rate Limit → Resolve Model          │
│       → Execute (+ Retry)                   │
│       → Account (durable write)             │
│       → Respond                             │
└────────────────┬────────────────────────────┘
                 │
         ┌───────┴───────┐
         │               │
      Ollama          (future:
   (local/free)     OpenAI, Groq...)
         │
    PostgreSQL          Redis
  (accounting        (auth cache,
    ledger)          rate limits)
```

### Key Design Decisions

| Decision | What & Why |
|---|---|
| **Strong accounting consistency** | A `200 OK` is never returned unless a durable accounting row exists in PostgreSQL. If the write fails, COALAI returns `503` rather than silently losing usage data. |
| **Fail-open Redis** | Redis failures never block requests. Rate limiting, auth cache, and circuit breaker all degrade gracefully if Redis is down. |
| **Unified error taxonomy** | All internal errors are `COALAIError` with a typed `error_type`. Provider-specific exceptions never cross adapter boundaries. |
| **No fire-and-forget** | Streaming uses a two-phase write: `PENDING` before the first chunk, `CONFIRMED` after the stream ends. A reconciliation job covers the gap. |
| **No Admin HTTP API in MVP** | Tenants and keys are provisioned via CLI script. Reduces attack surface and scope for Milestone 1. |

---

## Project Structure

```
COALAI/
├── src/coalai/
│   ├── config.py                  # All config from env vars (pydantic-settings)
│   ├── main.py                    # FastAPI app factory + lifespan
│   ├── models/
│   │   ├── contracts.py           # Domain types: Message, TenantContext, ExecutionResult...
│   │   └── errors.py              # COALAIError taxonomy
│   ├── auth/
│   │   └── service.py             # API key verification (Redis cache → Postgres fallback)
│   ├── cache/
│   │   └── redis_client.py        # Fail-open Redis wrapper
│   ├── db/
│   │   ├── models.py              # SQLAlchemy 2.0 ORM models
│   │   └── session.py             # Async engine + session management
│   ├── providers/
│   │   ├── base.py                # LLMProvider protocol
│   │   └── ollama.py              # Ollama adapter (streaming + non-streaming)
│   ├── reliability/
│   │   └── retry.py               # Exponential backoff with jitter
│   ├── accounting/
│   │   └── service.py             # Durable two-phase accounting writes
│   ├── gateway/
│   │   ├── schemas.py             # Pydantic request/response schemas
│   │   ├── pipeline.py            # Stage-based request pipeline
│   │   └── router.py              # FastAPI route handlers
│   └── observability/
│       └── logging.py             # structlog JSON logging
├── migrations/                    # Alembic async migrations
├── scripts/
│   └── provision.py               # CLI: create tenants + issue API keys
├── tests/
│   ├── unit/                      # 28 test cases, no infrastructure required
│   └── integration/               # Full Iron Path end-to-end tests
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Quickstart

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose
- Python 3.11+

### 1. Start infrastructure

```bash
docker compose up -d postgres redis ollama
```

### 2. Pull a model into Ollama

```bash
# Lightweight model for dev (~2GB)
docker exec coalai_ollama ollama pull llama3.2:3b

# Tiny model for CI / low-RAM machines (~400MB)
docker exec coalai_ollama ollama pull qwen2.5:0.5b
```

### 3. Set up the Python environment

```bash
pip install -e ".[dev]"
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env if your ports differ from defaults
```

### 5. Run database migrations

```bash
alembic upgrade head
```

### 6. Create a tenant and get an API key

```bash
python scripts/provision.py create --name "my-app"
```

Output:
```
✅  Tenant created successfully
   Tenant ID : xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

🔑  API Key (shown ONCE — store this securely):

   coal_sk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

   Authorization header:  Bearer coal_sk_xxxx...
```

### 7. Start the gateway

```bash
uvicorn coalai.main:app --reload --port 8000
```

### 8. Make a request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer coal_sk_your_key_here" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## API Reference

### `POST /v1/chat/completions`

OpenAI-compatible chat completion. Accepts the same request body as the OpenAI API.

**Request**

```json
{
  "model": "llama3.2:3b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is 2 + 2?"}
  ],
  "stream": false,
  "max_tokens": 256,
  "temperature": 0.7
}
```

**Response**

```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "model": "llama3.2:3b",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "2 + 2 = 4."},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 9,
    "total_tokens": 33
  },
  "coalai": {
    "request_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "trace_id": "abc123",
    "provider_used": "ollama",
    "model_used": "llama3.2:3b",
    "fallback_triggered": false,
    "attempt_number": 1,
    "cache_hit": false,
    "cache_type": null,
    "estimated_cost_usd": 0.0,
    "gateway_latency_ms": 12.4,
    "provider_latency_ms": 840.2,
    "total_latency_ms": 852.6,
    "ttft_ms": null
  }
}
```

The `coalai` field is included in every response and provides observability data for debugging and cost tracking.

### `GET /health`

Liveness probe. Returns `200` if the process is running.

### `GET /ready`

Readiness probe. Returns `200` only when both PostgreSQL and Redis are reachable.

```json
{"status": "ready", "postgres": true, "redis": true}
```

### Error Responses

All errors return a consistent JSON shape:

```json
{
  "error": {
    "error_type": "authentication_failed",
    "message": "Invalid API key",
    "request_id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
  }
}
```

| `error_type` | HTTP | Meaning |
|---|---|---|
| `authentication_failed` | 401 | Missing, malformed, revoked, or expired API key |
| `authorization_failed` | 403 | Tenant account inactive |
| `rate_limited` | 429 | Per-tenant RPM quota exceeded |
| `validation_error` | 422 | Invalid request body |
| `provider_timeout` | 504 | Ollama did not respond in time |
| `provider_rate_limited` | 429 | Provider returned 429 |
| `provider_server_error` | 502 | Provider returned 5xx |
| `accounting_write_failed` | 503 | Durable accounting write failed (accounting invariant) |
| `all_providers_failed` | 503 | All retry attempts exhausted |

---

## Configuration

All configuration is via environment variables. Copy `.env.example` to `.env`.

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://coalai:coalai@localhost:5432/coalai` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_DEFAULT_MODEL` | `llama3.2:3b` | Fallback model when none specified |
| `PROVIDER_TIMEOUT_SECONDS` | `30.0` | Provider call timeout |
| `PROVIDER_STREAMING_TIMEOUT_SECONDS` | `300.0` | Streaming timeout |
| `RETRY_MAX_ATTEMPTS` | `3` | Max retry attempts per request |
| `API_KEY_CACHE_TTL_SECONDS` | `300` | Redis TTL for verified key cache |
| `APP_DEBUG` | `false` | Enables SQL logging and `/docs` |

---

## Provisioning CLI

```bash
# Create a new tenant + API key
python scripts/provision.py create --name "my-app" --rpm 120 --rpd 50000

# List all tenants
python scripts/provision.py list

# Issue a new key for an existing tenant
python scripts/provision.py issue-key --tenant-id <uuid> --label "prod-server"
```

---

## Running Tests

```bash
# Unit tests — no infrastructure needed
pytest tests/unit/ -v

# Integration tests — requires Postgres + Redis (docker compose up -d)
pytest tests/integration/ -v -m integration

# All tests with coverage
pytest --cov=src/coalai --cov-report=term-missing
```

---

## Running with Docker Compose (full stack)

```bash
# Start everything
docker compose up -d

# Apply migrations
docker compose exec app alembic upgrade head

# Create a tenant
docker compose exec app python scripts/provision.py create --name "demo"

# Tail logs
docker compose logs -f app
```

---

## Roadmap

| Milestone | Status | Description |
|---|---|---|
| **M1 — Iron Path** | ✅ **Complete** | Auth, Ollama provider, durable accounting, retry, gateway |
| M2 — Observability | 🔜 | Prometheus metrics, OTEL tracing, Grafana dashboard |
| M3 — Exact Cache | 🔜 | Redis exact-match response caching |
| M4 — Semantic Cache | 🔜 | Embedding-based semantic cache with pgvector |
| M5 — Multi-Provider | 🔜 | Groq, OpenAI, Anthropic adapters + intelligent routing |
| M6 — Circuit Breaker | 🔜 | Per-provider circuit breaker with Redis state |
| M7 — Budget Control | 🔜 | Monetary budget enforcement per tenant |
| M8 — Admin API | 🔜 | HTTP control plane for tenant/key management |

---

## License

MIT
