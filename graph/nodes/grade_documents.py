from typing import Any, Dict

from graph.agent_builder import get_agent_builder
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    sources = state.get("sources", [])

    builder = get_agent_builder()

    filtered_docs = []
    web_search = False
    for d in documents:
        score = builder.retrieval_grader.run(
            question=question, document=d.page_content
        )
        grade = score.binary_score
        if grade and grade.lower() == "yes":
            print("---Grade: Document is RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---Grade: Document is NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search, "sources": sources}
