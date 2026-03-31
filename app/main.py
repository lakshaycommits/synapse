# python inbuilt imports
import asyncio
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any
from contextlib import asynccontextmanager
import os

# python packages imports
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends
from langchain_core.globals import set_llm_cache
from langchain_community.cache import RedisSemanticCache
from dotenv import load_dotenv
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

    set_llm_cache(RedisSemanticCache(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        embedding=app.state.embeddings.instance()
    ))

    yield
    app.state.qdrant._close_qrant_client()
    await producer.stop()

app = FastAPI(title="Synapse", lifespan = lifespan)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

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
async def agent_query(
    body: QueryRequest,
    graph: Annotated[Any, Depends(get_graph)],
):
    def _invoke():
        return graph.invoke({"query": body.query})

    result = await asyncio.to_thread(_invoke)
    return result
