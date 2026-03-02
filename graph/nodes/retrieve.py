from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    sources = []
    for doc in documents:
        if hasattr(doc, 'metadata') and 'source' in doc.metadata:
            sources.append(doc.metadata['source'])
    return {"documents": documents, "question": question, "sources": sources}
