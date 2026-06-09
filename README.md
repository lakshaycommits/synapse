# Synapse

**AI-powered engineering assistant** — multi-agent RAG pipeline that retrieves context from your documents, re-ranks it with a cross-encoder, and streams a grounded answer token-by-token.

[![Live Demo](https://img.shields.io/badge/demo-live%20on%20Render-46e3b7?style=flat-square)](https://synapse-nt8k.onrender.com/)
[![Python](https://img.shields.io/badge/python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-multi--agent-1C3C3C?style=flat-square)](https://langchain-ai.github.io/langgraph)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

---

## What it does

Engineering knowledge is fragmented across codebases, runbooks, and issue trackers. Synapse gives you a single natural language interface to query all of it:

- Upload your docs (PDF, Markdown, plain text)
- Ask a question in plain English
- Get a precise, grounded answer streamed in real-time — sourced from your documents, or the web as a fallback

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────┐
│  Planner                                 │
│  • Rewrites query for vector search      │
│  • Classifies route (index/web/general)  │
│  — single LLM call, no redundant hop —  │
└──────────────┬───────────────────────────┘
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
    index     web    general
       │       │        │
  Qdrant   Tavily    Groq LLM
  (top-10)  (top-3)  (direct)
       │
       ▼
┌──────────────────────────────────────────┐
│  Reflect  (Cross-Encoder Re-ranker)      │
│  • Scores all retrieved chunks vs query  │
│  • Keeps top-4 by relevance              │
│  • Routes to web fallback if score poor  │
│    (index route: honest "not found")     │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Response                                │
│  • Answers strictly from context         │
│  • Streams tokens via SSE                │
└──────────────────────────────────────────┘
```

### Key design decisions

| Decision | Why |
|---|---|
| Planner merges classification + query rewrite into one LLM call | Eliminates a redundant round-trip; structured text output (`ROUTE:` / `QUERY:`) parsed with zero-shot parsing |
| Cross-encoder re-ranking (ms-marco-MiniLM-L-6-v2) instead of LLM reflection | Better relevance signal at zero added inference cost; threshold-based quality gate |
| Index route never falls back to web search | Web search returns unrelated results for document-specific queries; honest "not found" is more useful |
| HuggingFace Inference API for embeddings | No local model loaded — saves ~250 MB RAM, critical for free-tier deployment |
| `ENABLE_RERANKING` env flag | Cross-encoder disabled on Render free tier (512 MB limit); enabled locally via `requirements-dev.txt` |
| FastAPI `BackgroundTasks` for ingestion | Upload API returns immediately; chunking + embedding runs async behind the response |
| Multi-stage Docker build (Node → Python) | React SPA served directly from FastAPI `StaticFiles` — no separate frontend hosting needed |

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (async, lifespan) |
| Agent orchestration | LangGraph |
| LLM | Groq (llama-3 family) |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| Re-ranking | sentence-transformers cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Vector database | Qdrant Cloud |
| Semantic cache | Redis |
| Web search | Tavily |
| Frontend | React 18 + Vite, react-markdown |
| Streaming | Server-Sent Events (SSE) via `graph.astream_events` |
| GitHub API | PyGithub |
| Containerization | Docker (multi-stage) + Docker Compose |
| Deployment | Render |

---

## Features

**Agent pipeline**
- Merged planner+router in a single LLM call — structured output parsed with zero-shot regex, no LLM function calling needed
- Cross-encoder re-ranking: retrieve top-10 from Qdrant, re-score every chunk against the query, pass top-4 to the response LLM
- Configurable relevance threshold (`RELEVANCE_THRESHOLD`) — controls the quality gate before web fallback
- Grounded responses — `response_node` is instructed to answer strictly from context, never from outside knowledge

**Ingestion & document management**
- Background ingestion via `FastAPI.BackgroundTasks` — upload returns in milliseconds
- SHA-256 content hashing for deduplication — same file uploaded twice doesn't create duplicate chunks
- Supports PDF, Markdown, and plain text
- Document index listing — `GET /rag/documents` returns all indexed files with chunk counts
- Targeted deletion — `DELETE /rag/documents?source=<name>` removes a document and every chunk that belongs to it from the vector store

**Infrastructure**
- Token streaming over SSE — `POST /agents/stream` emits `meta` → `token`... → `done` events; frontend renders progressively
- Semantic cache with Redis — near-duplicate queries skip the full pipeline
- Rate limiting with `slowapi` — 5 req/min per IP on query endpoints
- Structured logging with per-request UUID — every log line is traceable to a single request
- Health endpoint — `/health` reports live status of Qdrant and Redis

**GitHub integration**
- Manual sync via `POST /github/sync` — fetches all issues (with comments), PRs (with review comments + changed files), and code files from a branch; ingests each as a formatted markdown document
- Real-time updates via webhook — `POST /github/webhook` handles `issues`, `pull_request`, and `push` events; HMAC-SHA256 signature verification via `GITHUB_WEBHOOK_SECRET`
- File size guard — skips files over 500 KB; allowed extensions: `.py`, `.ts`, `.tsx`, `.js`, `.md`, `.yaml`, `.yml`
- Plugs directly into the existing ingestion pipeline — GitHub content goes through the same chunking, deduplication (SHA-256), and embedding path as manually uploaded files
- Dedicated GitHub tab in the UI with sync form and webhook setup instructions

**Frontend**
- Streaming answer with blinking cursor during inference
- Markdown rendering (headers, code blocks, tables, inline code) via react-markdown + remark-gfm
- Route badge shows how the query was routed (index / web / general)
- Drag-and-drop file upload with client-side validation
- Served as a single binary — React SPA bundled into `static/` at build time, served by FastAPI

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Live status of Qdrant and Redis |
| `POST` | `/rag/ingest` | Upload documents for background indexing |
| `GET` | `/rag/documents` | List all indexed documents with chunk counts |
| `DELETE` | `/rag/documents?source=<name>` | Remove a document and all its chunks from the index |
| `POST` | `/agents/query` | Submit a query, get full JSON response |
| `POST` | `/agents/stream` | Submit a query, get token-streamed SSE response |
| `POST` | `/github/sync` | Trigger a full sync of a GitHub repository |
| `POST` | `/github/webhook` | Receive real-time GitHub webhook events |

### `/agents/stream` — SSE event types

```
data: {"type": "meta",  "route": "index", "plan": "auth service 401 errors"}
data: {"type": "token", "token": "The "}
data: {"type": "token", "token": "auth service..."}
data: {"type": "done",  "context": ["...chunk 1...", "...chunk 2..."], "answer": "<full>"}
data: [DONE]
```

### `/agents/query` — JSON response

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

## Getting started

### Prerequisites

- Python 3.10+
- Node 18+
- Docker + Docker Compose
- API keys: Groq, Tavily, HuggingFace, Qdrant Cloud

### Environment variables

Create a `.env` file in the project root:

```env
# LLM
GROQ_API_KEY=
GROQ_LLM_MODEL=llama-3.3-70b-versatile
GROQ_LLM_TOOL_USE_MODEL=llama-3.3-70b-versatile

# Search
TAVILY_API_KEY=

# Embeddings
HUGGINGFACEHUB_API_TOKEN=
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Vector DB (Qdrant Cloud)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=
QDRANT_COLLECTION=synapse_docs

# Cache
REDIS_URL=redis://localhost:6379

# Re-ranking (set false to disable on memory-constrained environments)
ENABLE_RERANKING=true

# GitHub integration
GITHUB_TOKEN=ghp_...
GITHUB_WEBHOOK_SECRET=
```

### Run with Docker (recommended)

```bash
git clone https://github.com/your-username/synapse.git
cd synapse
cp .env.example .env   # fill in your keys
docker compose up --build
```

App available at `http://localhost:8000`.

### Run locally (infrastructure in Docker, code outside)

```bash
# Terminal 1 — infrastructure
docker compose up qdrant redis

# Terminal 2 — backend (with re-ranking enabled)
pip install -r requirements-dev.txt
uvicorn app.main:app --reload --port 8000

# Terminal 3 — frontend
cd frontend
npm install && npm run dev
```

Frontend dev server at `http://localhost:5173`, backend at `http://localhost:8000`.

---

## Project structure

```
synapse/
├── app/
│   ├── agents/
│   │   └── graph.py          # LangGraph pipeline (planner, retrieval, reflect, response)
│   ├── rag/
│   │   ├── ingest.py         # chunking, hashing, embedding, upsert
│   │   └── retriever.py      # Qdrant retriever
│   ├── utils/
│   │   ├── embeddings.py     # HuggingFace Inference API wrapper
│   │   ├── reranker.py       # optional cross-encoder (ENABLE_RERANKING flag)
│   │   ├── qdrantClient.py   # Qdrant client singleton
│   │   ├── variables.py      # RERANK_TOP_K, RELEVANCE_THRESHOLD
│   │   └── logger.py         # structured logger
│   ├── models/
│   │   └── request.py        # Pydantic request models
│   └── main.py               # FastAPI app, lifespan, routes, SSE endpoint
├── frontend/
│   └── src/
│       ├── App.jsx           # QueryPanel (streaming), IngestPanel
│       └── App.css
├── docs/
│   └── submit-no-match-contribution.md
├── Dockerfile                # multi-stage: Node (build) → Python (serve)
├── docker-compose.yml        # qdrant, redis, app
├── requirements.txt          # production deps (no sentence-transformers)
└── requirements-dev.txt      # + sentence-transformers for local re-ranking
```

---

## Useful commands

```bash
docker compose up --build          # start everything
docker compose up qdrant redis     # infrastructure only
docker compose down                # stop (data preserved)
docker compose down -v             # stop and wipe volumes
```

---

## Roadmap

- [ ] LangFuse observability — LLM traces, token usage, latency per node

---

## License

MIT
