import os
from typing import TypedDict, Literal

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langchain_community.tools.tavily_search import TavilySearchResults
load_dotenv()

from utils.variables import RERANK_TOP_K, RELEVANCE_THRESHOLD

class SynapseState(TypedDict):
    query: str
    plan: str
    route: str
    context: list
    context_quality: str
    answer: str


router_llm = ChatGroq(model=os.getenv("GROQ_LLM_MODEL"), api_key=os.getenv("GROQ_API_KEY"), temperature=0)
response_llm = ChatGroq(model=os.getenv("GROQ_LLM_TOOL_USE_MODEL"), api_key=os.getenv("GROQ_API_KEY"))


def planner_node(state: SynapseState) -> SynapseState:
    prompt = f"""You are a query planner. Given the user query, do two things:

        1. Classify it into exactly one category:
        - index: about specific documents, files, codebases, or named systems that may have been uploaded
        - web: needs real-time or recent information
        - general: common knowledge that needs no documents

        2. Rewrite it into a precise, keyword-rich search query (remove filler, keep technical terms and identifiers).

        User query: {state["query"]}

        Reply in exactly this format:
        ROUTE: <index|web|general>
        QUERY: <rewritten query>"""

    response = router_llm.invoke(prompt)
    lines = response.content.strip().splitlines()

    route = "general"
    plan = state["query"]
    for line in lines:
        if line.upper().startswith("ROUTE:"):
            val = line.split(":", 1)[1].strip().lower()
            if val in {"index", "web", "general"}:
                route = val
        elif line.upper().startswith("QUERY:"):
            plan = line.split(":", 1)[1].strip()

    return {"plan": plan, "route": route}


def route_decision(state: SynapseState) -> Literal["index", "web", "general"]:
    return state["route"]


def general_node(state: SynapseState) -> SynapseState:
    if len(state["query"].split()) <= 3:
        response = response_llm.invoke(state["query"], config={"cache": False})
    else:
        response = response_llm.invoke(state["query"])
    return {"answer": response.content}


def reflect_decision(state: SynapseState) -> Literal["response", "web_search"]:
    return "response" if state.get("context_quality") == "good" else "web_search"


def response_node(state: SynapseState) -> SynapseState:
    prompt = f"""Answer the query using the context below.
        Context: {state.get("context", [])}
        Query: {state["query"]}
        Give a precise, grounded answer."""
    response = response_llm.invoke(prompt)
    return {"answer": response.content}


def build_graph(retriever, reranker):
    graph = StateGraph(SynapseState)
    search_tool = TavilySearchResults(max_results=3)
    cross_encoder = reranker.instance()

    def retrieval_node(state: SynapseState) -> SynapseState:
        retrieval_query = state.get("plan") or state["query"]
        docs = retriever.invoke(retrieval_query)
        return {"context": [d.page_content for d in docs]}

    def reflect_node(state: SynapseState) -> SynapseState:
        context = state.get("context", [])
        if not context:
            return {"context_quality": "poor", "context": []}
        pairs = [[state["query"], chunk] for chunk in context]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(zip(scores, context), key=lambda x: x[0], reverse=True)
        top_chunks = [chunk for _, chunk in ranked[:RERANK_TOP_K]]
        quality = "good" if ranked[0][0] >= RELEVANCE_THRESHOLD else "poor"
        return {"context": top_chunks, "context_quality": quality}

    def web_search_node(state: SynapseState) -> SynapseState:
        results = search_tool.invoke(state["query"])
        return {"context": [r["content"] for r in results]}

    graph.add_node("planner", planner_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("general", general_node)
    graph.add_node("response", response_node)

    graph.add_edge(START, "planner")

    graph.add_conditional_edges("planner", route_decision, {
        "index": "retrieval",
        "web": "web_search",
        "general": "general",
    })

    graph.add_edge("retrieval", "reflect")
    graph.add_conditional_edges("reflect", reflect_decision, {
        "response": "response",
        "web_search": "web_search",
    })

    graph.add_edge("web_search", "response")
    graph.add_edge("response", END)
    graph.add_edge("general", END)

    return graph.compile()
