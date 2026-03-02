from typing import Any, Dict

from graph.agent_builder import get_agent_builder
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATING---")
    question = state["question"]
    documents = state["documents"]

    builder = get_agent_builder()
    generation = builder.generation.run(question=question, context=documents)
    
    sources = state.get("sources", [])
    
    final_report = f"""# Report

{generation}

## Sources

"""
    if sources:
        for i, source in enumerate(sources, 1):
            final_report += f"{i}. {source}\n"
    else:
        final_report += "No sources available.\n"
    
    return {"documents": documents, "question": question, "generation": final_report, "sources": sources}
