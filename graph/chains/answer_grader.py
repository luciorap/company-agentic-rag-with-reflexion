from typing import TYPE_CHECKING

from langchain_core.prompts import ChatPromptTemplate

if TYPE_CHECKING:
    from langchain_groq import ChatGroq

from pydantic import BaseModel, Field


class GradeAnswer(BaseModel):
    """Binary score for assessing if answer addresses the question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


class AnswerGraderAgent:
    """Agent for grading if an answer addresses the question."""

    def __init__(self, llm: "ChatGroq"):
        """Initialize with a shared LLM instance.
        
        Args:
            llm: Shared ChatGroq instance.
        """
        self.llm = llm
        self._chain = None

    def build(self):
        """Build and return the answer grader chain."""
        structured_llm_grader = self.llm.with_structured_output(GradeAnswer)

        system = """You are a grader assessing whether an answer addresses / resolves a question 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        
        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
            ]
        )

        self._chain = answer_prompt | structured_llm_grader
        return self._chain

    def run(self, question: str, generation: str) -> GradeAnswer:
        """Run the answer grader agent."""
        if self._chain is None:
            self.build()
        return self._chain.invoke({"question": question, "generation": generation})


def create_answer_grader(llm: "ChatGroq") -> AnswerGraderAgent:
    """Factory function to create an answer grader agent with shared LLM."""
    return AnswerGraderAgent(llm=llm)
