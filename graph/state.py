from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
        sources: list of sources used
        messages: messages for reflection agent
        iteration_count: number of reflection iterations
    """

    question: str
    generation: str
    web_search: bool
    documents: List[Document]
    sources: List[str]
    messages: List[BaseMessage]
    iteration_count: int
