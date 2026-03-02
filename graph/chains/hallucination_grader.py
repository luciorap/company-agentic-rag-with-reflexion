from typing import TYPE_CHECKING, Any

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_groq import ChatGroq

from pydantic import BaseModel, Field


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class HallucinationGraderAgent:
    """Agent for grading if generation is grounded in retrieved facts."""

    def __init__(self, llm: "ChatGroq"):
        """Initialize with a shared LLM instance.
        
        Args:
            llm: Shared ChatGroq instance.
        """
        self.llm = llm
        self._chain = None

    def build(self):
        """Build and return the hallucination grader chain."""
        structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)

        system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
            ]
        )

        self._chain = hallucination_prompt | structured_llm_grader
        return self._chain

    def run(self, documents: Any, generation: str) -> GradeHallucinations:
        """Run the hallucination grader agent."""
        if self._chain is None:
            self.build()
        return self._chain.invoke({"documents": documents, "generation": generation})


def create_hallucination_grader(llm: "ChatGroq") -> HallucinationGraderAgent:
    """Factory function to create a hallucination grader agent with shared LLM."""
    return HallucinationGraderAgent(llm=llm)
