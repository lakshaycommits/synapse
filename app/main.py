import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from rag.ingest import Ingestion
from rag.retriever import get_retriever
from utils.embeddings import embeddings

app = FastAPI(title="Synapse")

class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Question to run against the vector store")

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
async def retrieve(body: RetrieveRequest):
    retriever = get_retriever(embeddings.instance())
    docs = retriever.invoke(body.query)
    return {
        "query": body.query,
        "chunks": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
    }
