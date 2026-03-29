# python inbuilt imports
import asyncio
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any
from contextlib import asynccontextmanager

# python packages imports
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends

# graph imports
from agents.graph import build_graph

# rag imports
from rag.ingest import Ingestion
from rag.retriever import create_retriever

# utils imports
from utils.qdrantClient import qdrantClient
from utils.dependencies import get_graph, get_qdrant, get_retriever, get_embeddings
from utils.embeddings import Embeddings

# models imports
from models.request import QueryRequest

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.graph = build_graph()
    app.state.qdrant = qdrantClient()
    app.state.embeddings = Embeddings()
    app.state.retriever = create_retriever(app.state.embeddings.instance(), app.state.qdrant)
    yield
    app.state.qdrant._close_qrant_client()

app = FastAPI(title="Synapse", lifespan = lifespan)

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/rag/ingest")
async def rag_ingest(
    files: Annotated[list[UploadFile], File(description="PDF, .txt, or .md")],
    qdrant: Annotated[qdrantClient, Depends(get_qdrant)],
    embeddings: Annotated[Embeddings, Depends(get_embeddings)],
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

        chunks = Ingestion.ingest(tmp_paths, qdrant, embeddings)
        return {
            "chunks_indexed": chunks,
            "filenames": [f.filename for f in files],
        }

    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)

@app.post("/retrieve")
async def retrieve(
    body: QueryRequest,
    retriever: Annotated[Any, Depends(get_retriever)],
):
    docs = retriever.invoke(body.query)
    return {
        "query": body.query,
        "chunks": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
    }


@app.post("/agents/query")
async def agent_query(
    body: QueryRequest,
    graph: Annotated[Any, Depends(get_graph)],
):

    """Run the router graph (classifies query as index / web / general)."""

    def _invoke():
        return graph.invoke({"query": body.query})

    result = await asyncio.to_thread(_invoke)
    return result
