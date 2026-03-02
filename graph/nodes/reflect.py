from typing import Any, Dict, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from graph.chains.reflection_chains import first_responder, revisor
from graph.state import GraphState


MAX_ITERATIONS = 2


def draft_node(state: GraphState) -> Dict[str, Any]:
    """Draft the initial response using reflection agent."""
    print("---DRAFTING RESPONSE---")
    question = state["question"]
    
    messages = state.get("messages", [])
    if not messages:
        messages = [
            SystemMessage(
                content="Act as a Senior B2B Sales Intelligence Analyst and Market Researcher. "
                "Your goal is to produce actionable commercial profiles for the Company. "
                "For every research request, structure your response such as Company Identity, "
                "Commercial Value Prop, Market Context, Sales Triggers & Financials, Qualification. "
                "Provide detailed answers with citations."
            ),
            HumanMessage(content=question),
        ]
    
    response = first_responder.invoke({"messages": messages})
    
    return {
        "messages": messages + [response],
        "generation": response.tool_calls[0]["args"].get("answer", "") if response.tool_calls else str(response.content),
        "iteration_count": 0,
    }


def revise_node(state: GraphState) -> Dict[str, Any]:
    """Revise the answer based on tool results."""
    print("---REVISING RESPONSE---")
    messages = state.get("messages", [])
    
    response = revisor.invoke({"messages": messages})
    
    iteration_count = state.get("iteration_count", 0) + 1
    
    return {
        "messages": messages + [response],
        "generation": response.tool_calls[0]["args"].get("answer", "") if response.tool_calls else str(response.content),
        "iteration_count": iteration_count,
    }


def event_loop(state: GraphState) -> Literal["execute_tools", "end"]:
    """Determine whether to continue or end based on iteration count."""
    print("---CHECKING ITERATIONS---")
    iteration_count = state.get("iteration_count", 0)
    
    if iteration_count >= MAX_ITERATIONS:
        print(f"---MAX ITERATIONS REACHED ({MAX_ITERATIONS})---")
        return "end"
    
    return "execute_tools"
