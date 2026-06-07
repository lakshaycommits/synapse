# Synapse

An AI-powered engineering assistant that answers developer queries using a multi-agent RAG pipeline backed by a production-grade async infrastructure.

---

## Overview

Engineering knowledge is fragmented across codebases, docs, issue trackers, and the web. Synapse provides a single natural language interface that retrieves the right context from the right source and generates a precise, grounded answer — not a generic LLM response.

---

## Architecture

### Core Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (async) |
| Agent Orchestration | LangGraph |
| Vector Database | Qdrant |
| LLM Inference | Groq |
| Re-ranking | sentence-transformers (cross-encoder/ms-marco-MiniLM-L-6-v2) |
| Semantic Cache | Redis |
| Frontend | React + Vite |
| Containerization | Docker Compose |

---

### Agent Graph

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│ Planner Node                        │
│ • Rewrites query for retrieval      │
│ • Classifies route in one LLM call  │
└────────────────┬────────────────────┘
                 │
        ┌────────┼────────┐
        │        │        │
      index     web    general
        │        │        │
        ▼        ▼        ▼
   Retrieval  Web Search  LLM
   (Qdrant)   (Tavily)   Direct
      │
      ▼
┌─────────────────────────────────────┐
│ Reflect Node (Cross-Encoder)        │
│ • Scores top-10 chunks vs query     │
│ • Keeps top-4 by relevance score    │
│ • Falls back to web if score < -1.0 (web route only) │
└────────────────┬────────────────────┘
                 │
            ┌────┴────┐
          good       poor
            │           │
            ▼           ▼
        Response    Web Search
         Node        → Response
```

**Design decisions:**
- Planner and router merged into a single LLM call using structured text output (TOON format) — eliminates a redundant round-trip
- Cross-encoder re-ranking replaces an LLM-based reflection step — no extra inference cost, better relevance scoring
- Short queries (≤ 3 words) bypass the semantic cache to prevent false cache hits from overly similar embeddings

---

### Ingestion Pipeline

```
File Upload (PDF / .txt / .md)
    │
    ▼
FastAPI → BackgroundTask
                │
      ┌─────────┴──────────┐
      │  Deduplication     │
      │  (SHA-256 hash)    │
      └─────────┬──────────┘
                │
      Chunk + Embed + Store
                │
                ▼
             Qdrant
```

Files are saved on upload and ingested in a FastAPI BackgroundTask — the API returns immediately without blocking on chunking or embedding.

---

## Features

- **Optimized multi-agent graph** — Planner (classify + rewrite) → Retrieval → Cross-encoder reflect → Response, with web fallback on poor retrieval quality
- **Cross-encoder re-ranking** — retrieves top-10 chunks, re-ranks with a cross-encoder, passes top-4 to the response node — better answer quality without extra LLM calls
- **Async ingestion pipeline** — FastAPI BackgroundTasks decouples upload from indexing; the API returns immediately
- **Duplicate detection** — SHA-256 content hashing prevents re-indexing identical chunks across uploads
- **Semantic caching** — Redis-backed cache deduplicates LLM calls for semantically similar queries
- **Web search fallback** — Tavily search for real-time queries or when vector retrieval quality is insufficient
- **Rate limiting** — 5 requests/minute per IP on the query endpoint
- **Structured logging** — unique request ID propagated across all log lines for full request traceability
- **Health endpoint** — `/health` reports live status of Qdrant, Redis, and Kafka
- **React frontend** — query interface with route badge, retrieval sources, and async file upload with drag-and-drop
- **Docker Compose** — single command brings up all services (API, consumer, Qdrant, Kafka, Redis)
- **Hot reload** — source code mounted as volumes; `uvicorn --reload` picks up changes without rebuilds

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Live status of Qdrant, Redis, Kafka |
| `POST` | `/rag/ingest` | Upload documents (PDF, .txt, .md) for background indexing |
| `POST` | `/agents/query` | Submit a natural language query |

### Query response shape

```json
{
  "query": "Why is the auth service returning 401s?",
  "plan": "auth service 401 errors root cause",
  "route": "index",
  "context": ["...chunk 1...", "...chunk 2..."],
  "context_quality": "good",
  "answer": "The auth service is returning 401s because..."
}
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker + Docker Compose
- API keys: Groq, Tavily

### Environment variables

```env
GROQ_API_KEY=
GROQ_LLM_MODEL=
GROQ_LLM_TOOL_USE_MODEL=
TAVILY_API_KEY=
EMBEDDING_MODEL=
QDRANT_COLLECTION=
REDIS_URL=redis://localhost:6379
```

### Run everything with Docker

```bash
git clone https://github.com/your-username/synapse.git
cd synapse
docker compose up --build
```

### Run infrastructure in Docker, backend locally

```bash
# Terminal 1 — infrastructure
docker compose up qdrant redis

# Terminal 2 — backend
pip install -r requirements.txt
uvicorn app.main:app --reload

# Terminal 3 — frontend
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`, backend at `http://localhost:8000`.

---

## Useful commands

| Command | Description |
|---|---|
| `docker compose up --build` | Start all services |
| `docker compose up qdrant redis` | Start infrastructure only |
| `docker compose down` | Stop all services (data preserved) |
| `docker compose down -v` | Stop and wipe all volumes |

---

## Planned

- GitHub integration — auto-ingest PRs, issues, and code from connected repositories
- CI/CD pipeline (GitHub Actions)
- Observability — LangFuse traces + Prometheus/Grafana dashboards

---

## License

MIT
