from fastapi import Request

def get_graph(request: Request):
    return request.app.state.graph

def get_retriever(request: Request):
    return request.app.state.retriever

def get_qdrant(request: Request):
    return request.app.state.qdrant

def get_embeddings(request: Request):
    return request.app.state.embeddings

def get_producer(request: Request):
    return request.app.state.producer
