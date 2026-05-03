import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

class SynapseState(TypedDict):
    query: str
    route: str
    context: list
    answer: str
    plan: str

router_llm = ChatGroq(model=os.getenv("GROQ_LLM_MODEL"), api_key=os.getenv("GROQ_API_KEY"), temperature=0)
response_llm = ChatGroq(model=os.getenv("GROQ_LLM_TOOL_USE_MODEL"), api_key=os.getenv("GROQ_API_KEY"))

def router_node(state: SynapseState) -> SynapseState:
    prompt = f"""You are a query classifier. Classify the query below into exactly one category:

    - "index": query is about specific documents, files, codebases, or content that has been uploaded
    - "web": query needs real-time, recent, or live information
    - "general": query is about common knowledge that doesn't need any documents

    Important: If the query mentions a specific product, system, or named thing (like Synapse), classify as "index".

    Query: {state["query"]}

    Reply with only one word: index, web, or general."""

    response = router_llm.invoke(prompt)
    route = response.content.strip().lower()
    if route not in {"index", "web", "general"}:
        route = "general"
    return {"route": route}


def route_decision(state: SynapseState) -> Literal["retrieval", "general", "web"]:
    return state["route"]

def general_node(state: SynapseState) -> SynapseState:
    if len(state["query"].split()) <= 3:
        response = response_llm.invoke(state["query"], config={"cache": False})
    else:
        response = response_llm.invoke(state["query"])
    return {"answer": response.content}

def response_node(state: SynapseState) -> SynapseState:
    prompt = f"""Answer the query using the context below.
        Plan: {state.get("plan", "")}
        Context: {state.get("context", [])}
        Query: {state["query"]}
        Give a precise, grounded answer."""
    response = response_llm.invoke(prompt)
    return {"answer": response.content}

def planner_node(state: SynapseState) -> SynapseState:
    prompt = f"""You are a planner. Given this query, create a brief retrieval plan.
        Query: {state["query"]}
        Reply in one sentence: what to search for."""
    response = router_llm.invoke(prompt)
    return {"plan": response.content}

def build_graph(retriever):
    graph = StateGraph(SynapseState)
    search_tool = TavilySearchResults(max_results=3)

    def retreival_node(state: SynapseState) -> SynapseState:
        docs = retriever.invoke(state["query"])
        context = [d.page_content for d in docs]
        return {"context": context}

    def web_search_node(state: SynapseState) -> SynapseState:
        results = search_tool.invoke(state["query"])
        context = [r["content"] for r in results]
        return {"context": context}

    graph.add_node("planner", planner_node)
    graph.add_node("router", router_node)
    graph.add_node("retrieval", retreival_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)
    graph.add_node("web_search", web_search_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "router")

    graph.add_conditional_edges("router", route_decision, {
        "index": "retrieval",
        "web": "web_search",
        "general": "general"
    })

    graph.add_edge("retrieval", "response")
    graph.add_edge("web_search", "response")
    graph.add_edge("response", END)
    graph.add_edge("general", END)

    return graph.compile()
