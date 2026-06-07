import os
from dotenv import load_dotenv

load_dotenv()

class Reranker:
    def __init__(self):
        enabled = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
        if enabled:
            from sentence_transformers import CrossEncoder
            model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._instance = CrossEncoder(model)
        else:
            self._instance = None

    def instance(self):
        return self._instance

    def enabled(self) -> bool:
        return self._instance is not None
