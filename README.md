# Synapse

Engineering knowledge, unified and queryable.

---

## Overview

Synapse is an AI-powered engineering assistant that answers developer queries by aggregating context across:

* Codebases
* Documentation
* Issue tracking systems
* External sources

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

* Backend: FastAPI
* Orchestration: LangGraph
* Vector DB: Qdrant
* LLM Inference: Groq

---

### System Flow

1. User query
2. Planner agent decides:

   * Codebase search
   * Documentation retrieval
   * Ticket lookup
   * Web search
3. Retrieval agent fetches context
4. Response agent generates final answer

---

### Key Components

* Query Router

  * Classifies query type (code / docs / tickets / web)

* Vector Search

  * Embeddings stored in Qdrant

* Multi-Agent System

  * Planner → Retrieval → Response

* MCP Integration (planned)

  * GitHub
  * Confluence
  * Jira

---

## Features

* Context-aware answers
* Multi-source retrieval
* Tool-based reasoning (via MCP)
* Scalable ingestion pipeline (Kafka)
* Low-latency query handling (Redis caching planned)

---

## Project Phases

### Phase 1 — Core RAG (MVP)

* Basic ingestion pipeline
* Embedding + retrieval
* Query routing (index / general / web)
* FastAPI endpoints

---

### Phase 2 — Backend Scale

* Kafka-based ingestion pipeline
* Async document processing
* Redis caching
* Rate limiting

---

### Phase 3 — Multi-Agent + MCP

* Multi-agent orchestration
* Tool integration (GitHub, Jira, Confluence)
* Agent-to-agent communication

---

### Phase 4 — Production

* Docker Compose setup
* CI/CD pipeline
* Monitoring (Prometheus / Grafana / App Insights)

---

## Getting Started

### Prerequisites

* Python 3.10+
* Docker (optional but recommended)

---

### Setup

```bash
git clone https://github.com/your-username/synapse.git
cd synapse

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

### Run

```bash
uvicorn app.main:app --reload
```

---

## Example Query

```text
Why is the payment service failing in production?
```

Expected behavior:

* Checks recent Jira tickets
* Searches relevant services in codebase
* Retrieves logs / docs
* Returns a grounded explanation

---

## TODOs

### Phase 1 (Immediate)

* [ ] Implement document ingestion pipeline
* [ ] Add embedding generation
* [ ] Integrate Qdrant for vector storage
* [ ] Build query router (rule-based → ML later)
* [ ] Add basic web search fallback

---

### Phase 2 (Scale)

* [ ] Set up Kafka producer for document ingestion
* [ ] Build Kafka consumer for indexing
* [ ] Add async processing pipeline
* [ ] Implement Redis caching layer
* [ ] Add API rate limiting

---

### Phase 3 (Agents + MCP)

* [ ] Implement planner agent
* [ ] Implement retrieval agent
* [ ] Implement response agent
* [ ] Add MCP integration layer
* [ ] Connect GitHub as tool source
* [ ] Connect Jira API
* [ ] Connect Confluence API

---

### Phase 4 (Production)

* [ ] Dockerize all services
* [ ] Add docker-compose setup
* [ ] Set up CI/CD pipeline
* [ ] Add monitoring + logging
* [ ] Add structured tracing

---

## Known Gaps / Risks

* Query routing accuracy (critical bottleneck)
* Cross-source reasoning is hard (GitHub ↔ Jira linking)
* Latency can degrade with multiple tool calls
* Embedding quality impacts answer accuracy

---

## Contribution

* Open issues for bugs / feature requests
* Keep PRs small and focused
* Add tests where applicable

---

## License

MIT License (recommended)
