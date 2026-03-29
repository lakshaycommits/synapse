import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

load_dotenv()

class SynapseState(TypedDict):
    query: str
    route: str
    context: list
    answer: str

llm = ChatGroq(model=os.getenv("GROQ_LLM_MODEL"), api_key=os.getenv("GROQ_API_KEY"))

def router_node(state: SynapseState) -> SynapseState:
    prompt = f"""Classify this query into one of three categories:
        - "index": question about specific documents or codebases
        - "web": needs real-time or recent information  
        - "general": common knowledge question

        Query: {state["query"]}

        Reply with only one word: index, web, or general.
    """

    response = llm.invoke(prompt)
    route = response.content.strip().lower()
    if route not in {"index", "web", "general"}:
        route = "general"
    return {"route": route}

def build_graph():
    graph = StateGraph(SynapseState)
    graph.add_node("router", router_node)
    graph.add_edge(START, "router")
    graph.add_edge("router", END)
    return graph.compile()

synapse_graph = build_graph()
