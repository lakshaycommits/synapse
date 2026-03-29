from __future__ import annotations

import os
from pathlib import Path

from utils import qdrantClient
from utils.embeddings import Embeddings

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

os.environ.setdefault("USER_AGENT", "synapse-agent")

class Ingestion:

    @staticmethod
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

    @staticmethod
    def split_documents(docs):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        return splitter.split_documents(docs)

    @staticmethod
    def ingest(paths: list[Path], qdrant: qdrantClient, embeddings: Embeddings) -> int:
        print(f"Loaded {len(paths)} file(s)")

        docs = Ingestion.load_documents(paths)
        splits = Ingestion.split_documents(docs)
        vector_store = qdrant._get_vector_store(embeddings.instance())
        vector_store.add_documents(splits)

        print(f"Indexed {len(splits)} chunk(s) into Qdrant collection {qdrant._get_collection_name()!r}")
        print("Ingestion complete")
        return len(splits)
