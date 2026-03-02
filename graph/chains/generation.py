from typing import TYPE_CHECKING, Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_groq import ChatGroq


class GenerationAgent:
    """Agent for generating answers based on retrieved context."""

    def __init__(self, llm: "ChatGroq"):
        """Initialize with a shared LLM instance.
        
        Args:
            llm: Shared ChatGroq instance.
        """
        self.llm = llm
        self._chain = None

    def build(self):
        """Build and return the generation chain."""
        template = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:"""

        prompt = ChatPromptTemplate.from_template(template)
        self._chain = prompt | self.llm | StrOutputParser()
        return self._chain

    def run(self, question: str, context: Any) -> str:
        """Run the generation agent."""
        if self._chain is None:
            self.build()
        return self._chain.invoke({"question": question, "context": context})


def create_generation_agent(llm: "ChatGroq") -> GenerationAgent:
    """Factory function to create a generation agent with shared LLM."""
    return GenerationAgent(llm=llm)
