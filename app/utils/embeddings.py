from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

class Embeddings:
    def __init__(self):
        self._instance = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL"))

    def instance(self) -> HuggingFaceEmbeddings:
        return self._instance
