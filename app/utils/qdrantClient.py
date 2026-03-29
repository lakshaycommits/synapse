
import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore

class qdrantClient:
    def __init__(self):
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
        self._instance = QdrantClient(url=self.QDRANT_URL)

    def _ensure_collection(self, collection_name: str, embeddings) -> None:
        existing = {c.name for c in self._instance.get_collections().collections}
        if collection_name in existing:
            return
        dim = len(embeddings.embed_query("."))
        self._instance.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=dim,
                distance=qmodels.Distance.COSINE,
            ),
        )

    def _get_collection_name(self):
        return self.COLLECTION_NAME

    def _get_instance(self):
        return self._instance

    def _get_vector_store(self, embeddings) -> QdrantVectorStore:
        self._ensure_collection(self.COLLECTION_NAME, embeddings)
        return QdrantVectorStore(
            client=self._instance,
            collection_name=self.COLLECTION_NAME,
            embedding=embeddings,
        )

    def _close_qrant_client(self):
        self._get_instance().close()
