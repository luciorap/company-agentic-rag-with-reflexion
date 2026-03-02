from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATING---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    
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
