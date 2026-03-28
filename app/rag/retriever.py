from utils.qdrantClient import qdrant

def get_retriever(embeddings, k=4):
    vector_store = qdrant._get_vector_store(embeddings)
    return vector_store.as_retriever(search_kwargs={"k": k})
