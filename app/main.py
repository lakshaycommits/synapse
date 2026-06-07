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
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile, Depends
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()

# graph imports
from .agents.graph import build_graph

# rag imports
from .rag.ingest import Ingestion
from .rag.retriever import create_retriever

# utils imports
from .utils.qdrantClient import qdrantClient
from .utils.dependencies import get_graph, get_qdrant, get_embeddings
from .utils.embeddings import Embeddings
from .utils.reranker import Reranker
from .utils.logger import get_logger
logger = get_logger()

# models imports
from .models.request import QueryRequest


def _ingest_file(path: Path, qdrant: qdrantClient, embeddings: Embeddings):
    try:
        chunks = Ingestion.ingest([path], qdrant, embeddings)
        logger.info("Ingest succeeded: %s → %s chunks", path.name, chunks)
    except Exception:
        logger.exception("Error ingesting %s", path.name)
    finally:
        path.unlink(missing_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.qdrant = qdrantClient()
        app.state.embeddings = Embeddings()
        app.state.reranker = Reranker()
        app.state.retriever = create_retriever(app.state.embeddings.instance(), app.state.qdrant)
        app.state.graph = build_graph(app.state.retriever, app.state.reranker)
        app.state.redis_client = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
    except Exception as e:
        logger.exception("Startup initialization failed: %s", e)
        raise

    app.add_exception_handler(
        RateLimitExceeded,
        lambda request, exc: JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        ),
    )

    yield
    try:
        app.state.qdrant._close_qrant_client()
    except Exception:
        pass


app = FastAPI(title="Synapse", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    try:
        qdrant._get_collection_name()
        status["qdrant"] = "up"
    except Exception as e:
        logger.error(f"Qdrant health failed: {e}")
        status["qdrant"] = "down"

    try:
        app.state.redis_client.ping()
        status["redis"] = "up"
    except Exception as e:
        logger.error(f"Redis health failed: {e}")
        status["redis"] = "down"

    overall = "ok" if all(v == "up" for v in status.values()) else "degraded"

    response = {"status": overall, "services": status}

    if overall == "degraded":
        raise HTTPException(status_code=500, detail=response)
    return response


@app.post("/rag/ingest")
async def rag_ingest(
    files: Annotated[list[UploadFile], File(description="PDF, .txt, or .md")],
    background_tasks: BackgroundTasks,
    qdrant: Annotated[qdrantClient, Depends(get_qdrant)],
    embeddings: Annotated[Embeddings, Depends(get_embeddings)],
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    tmp_paths: list[Path] = []
    for upload in files:
        name = upload.filename or "upload"
        suffix = Path(name).suffix or ".txt"
        upload_dir = Path(os.getenv("UPLOAD_DIR", "/tmp"))
        upload_dir.mkdir(parents=True, exist_ok=True)
        tmp = NamedTemporaryFile(delete=False, suffix=suffix, dir=upload_dir)
        try:
            shutil.copyfileobj(upload.file, tmp)
        finally:
            tmp.close()
        upload.file.close()
        tmp_paths.append(Path(tmp.name))

    for path in tmp_paths:
        background_tasks.add_task(_ingest_file, path, qdrant, embeddings)

    return {
        "message": "files queued for ingestion",
        "filenames": [f.filename for f in files],
    }


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
