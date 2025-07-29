from langgraph.graph import StateGraph, START, END
from langgraph.graph.state import StateGraph

from flow.state import State
from flow.async_nodes import a_call_model, a_summarize_conversation, a_suggestion_for_user
from flow.nodes import call_model, summarize_conversation, suggestion_for_user
from flow.conditions import should_continue


def a_build_graph() -> StateGraph:
    """Builds and compiles the LangGraph StateGraph."""

    workflow = StateGraph(State)

    workflow.add_node("conversation", a_call_model)
    workflow.add_node("summarize_conversation", a_summarize_conversation)
    workflow.add_edge(START, "conversation")

    workflow.add_conditional_edges(
        "conversation",
        should_continue,
        {
            "summarize_conversation": "summarize_conversation",
            END: END
        }
    )
    workflow.add_edge("summarize_conversation", END)
    return workflow


def build_graph() -> StateGraph:
    """Builds and compiles the LangGraph StateGraph."""

    workflow = StateGraph(State)

    workflow.add_node("conversation", a_call_model)
    workflow.add_node("summarize_conversation", a_summarize_conversation)
    workflow.add_edge(START, "conversation")

    workflow.add_conditional_edges(
        "conversation",
        should_continue,
        {
            "summarize_conversation": "summarize_conversation",
            END: END
        }
    )
    workflow.add_edge("summarize_conversation", END)
    return workflow
