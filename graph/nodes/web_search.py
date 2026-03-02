from typing import Any, Dict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()
web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents")
        
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    
    sources = [result["url"] for result in tavily_results]
    
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    existing_sources = state.get("sources", [])
    all_sources = existing_sources + sources
    
    return {"documents": documents, "question": question, "sources": all_sources}


if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None}) # type: ignore
