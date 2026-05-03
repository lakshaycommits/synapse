
import os
from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import QdrantVectorStore
import time
import logging

logger = logging.getLogger(__name__)


class qdrantClient:
    def __init__(self, retries: int | None = None, delay: float | None = None):
        """Create a Qdrant client with retry/backoff on connection errors.

        Reads the following environment variables when appropriate:
        - QDRANT_URL
        - QDRANT_HOST (default: 'qdrant')
        - QDRANT_PORT (default: 6333)
        - QDRANT_CONNECT_RETRIES (optional override)
        - QDRANT_CONNECT_DELAY (optional override, seconds)
        """
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        host = os.getenv("QDRANT_HOST", "qdrant")
        port = os.getenv("QDRANT_PORT", "6333")
        self.COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")

        # allow env override for retries/delay
        try:
            env_retries = int(os.getenv("QDRANT_CONNECT_RETRIES", ""))
        except Exception:
            env_retries = None
        try:
            env_delay = float(os.getenv("QDRANT_CONNECT_DELAY", ""))
        except Exception:
            env_delay = None

        self.retries = retries if retries is not None else (env_retries if env_retries is not None else 10)
        self.delay = delay if delay is not None else (env_delay if env_delay is not None else 2.0)

        if not self.QDRANT_URL:
            self.QDRANT_URL = f"http://{host}:{port}"

        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                logger.info("Connecting to Qdrant (attempt %d/%d) at %s", attempt, self.retries, self.QDRANT_URL)
                self._instance = QdrantClient(url=self.QDRANT_URL)
                # quick health check
                self._instance.get_collections()
                logger.info("Connected to Qdrant at %s", self.QDRANT_URL)
                last_exc = None
                break
            except Exception as e:
                last_exc = e
                logger.warning("Qdrant connection attempt %d failed: %s", attempt, e)
                if attempt < self.retries:
                    time.sleep(self.delay * attempt)  # linear backoff

        if last_exc is not None:
            # raise an informative error for caller to handle
            raise ConnectionError(f"Failed to connect to Qdrant at {self.QDRANT_URL} after {self.retries} attempts: {last_exc}")

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
