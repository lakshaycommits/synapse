import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

os.environ.setdefault("USER_AGENT", "synapse-agent")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

def _make_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)

def _ensure_collection(client: QdrantClient, collection_name: str, embeddings) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if collection_name in existing:
        return
    dim = len(embeddings.embed_query("."))
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=dim,
            distance=qmodels.Distance.COSINE,
        ),
    )

def load_documents(paths: list[Path]) -> list:
    if not paths:
        raise ValueError("At least one file path is required.")
    docs = []
    for path in paths:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(PyPDFLoader(str(path)).load())
        elif suffix in {".txt", ".md", ".markdown"}:
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
        else:
            raise ValueError(
                f"Unsupported type {suffix!r} for {path.name}. "
                "Use .pdf, .txt, or .md."
            )
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)


def get_vector_store(embeddings):
    client = _make_qdrant_client()
    _ensure_collection(client, COLLECTION_NAME, embeddings)
    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings,
    )

def ingest(paths: list[Path]) -> int:
    print(f"Loaded {len(paths)} file(s)")

    docs = load_documents(paths)
    splits = split_documents(docs)
    vector_store = get_vector_store(embeddings)
    vector_store.add_documents(splits)

    print(f"Indexed {len(splits)} chunk(s) into Qdrant collection {COLLECTION_NAME!r}")
    print("Ingestion complete")
    return len(splits)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk local files and index embeddings into Qdrant."
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="PDF, .txt, or .md files to index",
    )
    args = parser.parse_args()
    ingest(args.files)
