# python inbuilt imports
import asyncio
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

# python packages imports
from fastapi import FastAPI, File, HTTPException, UploadFile

# graph imports
from agents.graph import synapse_graph

# rag imports
from rag.ingest import Ingestion
from rag.retriever import get_retriever

# utils imports
from utils.embeddings import embeddings

# models imports
from models.request import QueryRequest

app = FastAPI(title="Synapse")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/rag/ingest")
async def rag_ingest(
    files: Annotated[list[UploadFile], File(description="PDF, .txt, or .md")],
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

        chunks = Ingestion.ingest(tmp_paths)
        return {
            "chunks_indexed": chunks,
            "filenames": [f.filename for f in files],
        }
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)

@app.post("/retrieve")
async def retrieve(body: QueryRequest):
    retriever = get_retriever(embeddings.instance())
    docs = retriever.invoke(body.query)
    return {
        "query": body.query,
        "chunks": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
    }


@app.post("/agents/query")
async def agent_query(body: QueryRequest):
    """Run the router graph (classifies query as index / web / general)."""

    def _invoke():
        return synapse_graph.invoke({"query": body.query})

    result = await asyncio.to_thread(_invoke)
    return result
