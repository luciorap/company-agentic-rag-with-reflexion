from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_groq import ChatGroq

from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class RetrievalGraderAgent:
    """Agent for grading document relevance to a question."""

    def __init__(self, llm: "ChatGroq"):
        """Initialize with a shared LLM instance.
        
        Args:
            llm: Shared ChatGroq instance.
        """
        self.llm = llm
        self._chain = None

    def build(self):
        """Build and return the retrieval grader chain."""
        structured_llm_grader = self.llm.with_structured_output(GradeDocuments)

        system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        
        grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )

        self._chain = grade_prompt | structured_llm_grader
        return self._chain

    def run(self, question: str, document: str) -> GradeDocuments:
        """Run the retrieval grader agent."""
        if self._chain is None:
            self.build()
        return self._chain.invoke({"question": question, "document": document})


def create_retrieval_grader(llm: "ChatGroq") -> RetrievalGraderAgent:
    """Factory function to create a retrieval grader agent with shared LLM."""
    return RetrievalGraderAgent(llm=llm)
