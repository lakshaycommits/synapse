import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.rag.ingest import ingest

app = FastAPI(title="Synapse")

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

        chunks = ingest(tmp_paths)
        return {
            "chunks_indexed": chunks,
            "filenames": [f.filename for f in files],
        }
    finally:
        for p in tmp_paths:
            p.unlink(missing_ok=True)
