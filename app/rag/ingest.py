from __future__ import annotations

import os
from pathlib import Path

from utils import qdrantClient
from utils.embeddings import Embeddings
from utils.helper_functions import _get_doc_hash
from utils.logger import get_logger

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
        logger = get_logger(__name__)
        logger.info("Loaded %d file(s)", len(paths))

        docs = Ingestion.load_documents(paths)
        splits = Ingestion.split_documents(docs)

        new_splits = Ingestion.check_duplication(qdrant, splits)
        
        if not new_splits:
            logger.info("No new chunks to index")
            return 0

        vector_store = qdrant._get_vector_store(embeddings.instance())
        vector_store.add_documents(new_splits)

        logger.info(
            "Indexed %d chunk(s) into Qdrant collection %r",
            len(new_splits),
            qdrant._get_collection_name(),
        )
        logger.debug("Ingestion complete")
        return len(new_splits)

    @staticmethod
    def _get_existing_hashes(qdrant: qdrantClient) -> set:
        try:
            results = qdrant._get_instance().scroll(
                collection_name=os.getenv("QDRANT_COLLECTION"),
                with_payload=True,
                limit=10000
            )

            return {
                point.payload.get("metadata", {}).get("hash")
                for point in results[0]
                if point.payload.get("metadata", {}).get("hash")
            }
        except:
            return set()

    @staticmethod
    def check_duplication(qdrant, splits) -> []:
        existing = Ingestion._get_existing_hashes(qdrant)
        new_splits = []

        for doc in splits:
            doc_hash = _get_doc_hash(doc.page_content)
            if doc_hash not in existing:
                doc.metadata["hash"] = doc_hash
                new_splits.append(doc)

        return new_splits
