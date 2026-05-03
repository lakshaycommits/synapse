# Synapse

Engineering knowledge, unified and queryable.

---

## Overview

Synapse is an AI-powered engineering assistant that answers developer queries by aggregating context across codebases, documentation, issue tracking systems, and the web.

It uses Retrieval-Augmented Generation (RAG) and a multi-agent architecture to provide grounded, context-aware answers instead of generic LLM responses.

---

## Problem

Engineering knowledge is fragmented:

* Code → GitHub
* Docs → Confluence
* Issues → Jira
* Discussions → Slack / Web

Finding answers requires manual search across multiple tools.

---

## Solution

Synapse provides a single interface where:

* Engineers ask questions in natural language
* The system decides the best data source
* Relevant context is retrieved
* A precise, grounded answer is generated

---

## Architecture

### Core Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Orchestration | LangGraph |
| Vector DB | Qdrant |
| LLM Inference | Groq |
| Message Queue | Kafka (KRaft) |
| Caching | Redis (Semantic Cache) |
| Containerization | Docker Compose |

---

### System Flow

```
User Query
    │
    ▼
Planner Agent  ──→  creates a retrieval plan
    │
    ▼
Router Agent   ──→  classifies: index | web | general
    │
    ├── index   ──→  Qdrant vector retrieval  ──→  Response Agent
    ├── web     ──→  Tavily web search        ──→  Response Agent
    └── general ──→  LLM direct answer
```

---

## Features

### Implemented

* **Multi-agent graph** — Planner → Router → Retrieval/Web/General → Response (LangGraph)
* **RAG pipeline** — document ingestion (PDF, .txt, .md), chunking, embedding, and vector search (Qdrant)
* **Async ingestion** — Kafka producer/consumer pipeline; files are queued on upload and indexed in the background
* **Duplicate detection** — content hashing prevents re-indexing identical chunks
* **Semantic caching** — Redis-backed LLM response cache reduces redundant inference calls
* **Web search fallback** — Tavily search for real-time queries
* **Rate limiting** — 5 requests/minute on the query endpoint (slowapi)
* **Structured logging** — per-request IDs across all log lines
* **Health check** — `/health` endpoint reports live status of Qdrant, Redis, and Kafka
* **Docker Compose** — single command spins up all services (app, Qdrant, Kafka, Redis)

### Planned

* MCP tool integrations — GitHub, Jira, Confluence
* CI/CD pipeline
* Monitoring — Prometheus + Grafana

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health (Qdrant, Redis, Kafka) |
| `POST` | `/rag/ingest` | Upload documents (PDF, .txt, .md) for indexing |
| `POST` | `/agents/query` | Submit a natural language query |

---

## Getting Started

### Prerequisites

* Python 3.10+
* Docker + Docker Compose

### Run with Docker

```bash
git clone https://github.com/your-username/synapse.git
cd synapse
docker compose up --build
```

### Run locally

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

---

## Example Query

```text
Why is the payment service failing in production?
```

Expected behavior:

* Router classifies query as `index`
* Planner scopes the retrieval to relevant services
* Qdrant retrieves matching chunks from ingested codebases/docs
* Response agent returns a grounded explanation

---

## Known Limitations

* Query routing accuracy is the critical bottleneck — misclassification degrades answer quality
* Cross-source reasoning (e.g. linking a GitHub commit to a Jira ticket) is not yet supported
* Latency increases with multiple tool hops
* Embedding quality directly impacts retrieval relevance

---

## License

MIT
