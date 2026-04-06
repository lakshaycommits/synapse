# python inbuilt imports
import asyncio
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any
from contextlib import asynccontextmanager
import os
import uuid

# python packages imports
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, Depends
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
from fastapi.responses import JSONResponse
load_dotenv()

# graph imports
from agents.graph import build_graph

# rag imports
from rag.ingest import Ingestion
from rag.retriever import create_retriever

# utils imports
from utils.qdrantClient import qdrantClient
from utils.dependencies import get_graph, get_producer, get_qdrant, get_embeddings
from utils.embeddings import Embeddings
from utils.logger import get_logger
logger = get_logger()

# models imports
from models.request import QueryRequest

# kafka imports
from kafka.producer import Producer

@asynccontextmanager
async def lifespan(app: FastAPI):
    producer = Producer()
    await producer.start()
    app.state.producer = producer

    app.state.qdrant = qdrantClient()
    app.state.embeddings = Embeddings()
    app.state.retriever = create_retriever(app.state.embeddings.instance(), app.state.qdrant)
    app.state.graph = build_graph(app.state.retriever)
    app.state.redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

    app.add_exception_handler(
        RateLimitExceeded,
        lambda request, exc: JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        ),
    )

    set_llm_cache(RedisSemanticCache(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        embedding=app.state.embeddings.instance(),
        score_threshold=0.2
    ))

    yield
    app.state.qdrant._close_qrant_client()
    await producer.stop()


app = FastAPI(title="Synapse", lifespan = lifespan)
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.middleware("http")
async def log_requests(request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    logger.info(f"[{request_id}] {request.method} {request.url}")

    response = await call_next(request)

    logger.info(f"[{request_id}] {response.status_code}")

    return response

@app.get("/health")
async def health_check(
    qdrant: Annotated[qdrantClient, Depends(get_qdrant)]
):
    status = {}

    # Qdrant
    try:
        qdrant._get_collection_name()
        status["qdrant"] = "up"
    except Exception as e:
        logger.error(f"Qdrant health failed: {e}")
        status["qdrant"] = "down"

    # Redis
    try:
        app.state.redis_client.ping()
        status["redis"] = "up"
    except Exception as e:
        logger.error(f"Redis health failed: {e}")
        status["redis"] = "down"

    # Kafka (basic check)
    try:
        producer = app.state.producer
        await producer._producer.client.bootstrap()
        if producer:
            status["kafka"] = "up"
        else:
            status["kafka"] = "down"
    except Exception as e:
        logger.error(f"Kafka health failed: {e}")
        status["kafka"] = "down"

    overall = "ok" if all(v == "up" for v in status.values()) else "degraded"

    response = {
        "status": overall,
        "services": status
    }

    if overall == "degraded":
        raise HTTPException(status_code=500, detail=response)
    return response


@app.post("/rag/ingest")
async def rag_ingest(
    files: Annotated[list[UploadFile], File(description="PDF, .txt, or .md")],
    qdrant: Annotated[qdrantClient, Depends(get_qdrant)],
    embeddings: Annotated[Embeddings, Depends(get_embeddings)],
    producer: Annotated[Producer, Depends(get_producer)],
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    tmp_paths: list[Path] = []
    try:
        for upload in files:
            name = upload.filename or "upload"
            suffix = Path(name).suffix or ".txt"
            tmp = NamedTemporaryFile(delete=False, suffix=suffix)
            try:
                shutil.copyfileobj(upload.file, tmp)
            finally:
                tmp.close()
            upload.file.close()
            tmp_paths.append(Path(tmp.name))

        for path in tmp_paths:
            await producer.publish_ingest_event(str(path), path.name)

        return {
            "message": "files queued for ingestion",
            "filenames": [f.filename for f in files],
        }

    finally:
        pass

@app.post("/agents/query")
@limiter.limit("5/minute")
async def agent_query(
    body: QueryRequest,
    request: Request,
    graph: Annotated[Any, Depends(get_graph)],
):
    def _invoke():
        return graph.invoke({"query": body.query})

    result = await asyncio.to_thread(_invoke)
    return result
