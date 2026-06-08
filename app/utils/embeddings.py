import os
from typing import List
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.embeddings import Embeddings as LangChainEmbeddings

load_dotenv()


class Embeddings(LangChainEmbeddings):
    def __init__(self):
        model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        # ensure full model path is always used
        if "/" not in model:
            model = f"sentence-transformers/{model}"
        self._model = model
        self._client = InferenceClient(token=os.getenv("HUGGINGFACEHUB_API_TOKEN"))

    def embed_query(self, text: str) -> List[float]:
        result = self._client.feature_extraction(text, model=self._model)
        return result.tolist() if hasattr(result, "tolist") else list(result)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

    def instance(self) -> "Embeddings":
        return self
