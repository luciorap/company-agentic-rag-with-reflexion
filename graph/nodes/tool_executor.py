from dotenv import load_dotenv

load_dotenv()

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode

from graph.schemas import AnswerQuestion, ReviseAnswer

tavily_tool = TavilySearch(max_results=5)


def run_queries(search_queries: list[str] = None, **kwargs):
    """Run the generated queries."""
    if search_queries is None:
        search_queries = []
    return tavily_tool.batch([{"query": query} for query in search_queries])


execute_tools = ToolNode(
    [
        StructuredTool.from_function(
            func=run_queries, 
            name="AnswerQuestion",
            description="Run search queries to research information"
        ),
    ]
)
