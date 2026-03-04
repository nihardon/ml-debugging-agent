"""LangGraph DAG wiring all 3 nodes: parse → retrieve → advise."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from backend.agents.advisor import advise_node
from backend.agents.parser import parse_node
from backend.agents.retriever import retrieve_node
from backend.models import GraphState


def _sync_parse(state: GraphState) -> GraphState:
    return parse_node(state)


def _sync_retrieve(state: GraphState) -> GraphState:
    return retrieve_node(state)


def build_graph():
    """Build and compile the LangGraph StateGraph."""
    builder = StateGraph(GraphState)

    builder.add_node("parse", _sync_parse)
    builder.add_node("retrieve", _sync_retrieve)
    builder.add_node("advise", advise_node)

    builder.add_edge(START, "parse")
    builder.add_edge("parse", "retrieve")
    builder.add_edge("retrieve", "advise")
    builder.add_edge("advise", END)

    return builder.compile()


# Module-level compiled graph — imported by api.py
graph = build_graph()
