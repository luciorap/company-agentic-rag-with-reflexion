from typing import TYPE_CHECKING, Literal

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_groq import ChatGroq

from pydantic import BaseModel, Field


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


class RouterAgent:
    """Agent for routing user questions to the appropriate datasource."""

    def __init__(self, llm: "ChatGroq"):
        """Initialize with a shared LLM instance.
        
        Args:
            llm: Shared ChatGroq instance.
        """
        self.llm = llm
        self._chain = None

    def build(self):
        """Build and return the router chain."""
        structured_llm_router = self.llm.with_structured_output(RouteQuery)

        system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""
        
        route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "{question}"),
            ]
        )

        self._chain = route_prompt | structured_llm_router
        return self._chain

    def run(self, question: str) -> RouteQuery:
        """Run the router agent."""
        if self._chain is None:
            self.build()
        return self._chain.invoke({"question": question})


def create_router(llm: "ChatGroq") -> RouterAgent:
    """Factory function to create a router agent with shared LLM."""
    return RouterAgent(llm=llm)
