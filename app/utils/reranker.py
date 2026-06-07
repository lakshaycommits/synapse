import os
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

class Reranker:
    def __init__(self):
        model = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self._instance = CrossEncoder(model)

    def instance(self) -> CrossEncoder:
        return self._instance
