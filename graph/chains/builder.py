from typing import Optional

from langchain_groq import ChatGroq

from graph.chains.answer_grader import AnswerGraderAgent
from graph.chains.generation import GenerationAgent
from graph.chains.hallucination_grader import HallucinationGraderAgent
from graph.chains.reflection_chains import ReflectionAgent
from graph.chains.retrieval_grader import RetrievalGraderAgent
from graph.chains.router import RouterAgent


class AgentBuilder:
    """Builder class for creating and configuring all agents in the graph.
    
    Usage:
        builder = AgentBuilder(model="llama-3.3-70b-versatile", temperature=0).build()
        builder.router.run(question="...")
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_TEMPERATURE = 0

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
    ):
        self.model = model
        self.temperature = temperature
        self._llm: Optional[ChatGroq] = None
        self._built = False

    def build(self) -> "AgentBuilder":
        """Build all agents with a single LLM instance.
        
        Returns:
            Self for chaining.
        """
        if self._built:
            return self
            
        self._llm = ChatGroq(
            model=self.model, 
            temperature=self.temperature
        )

        self._router = RouterAgent(llm=self._llm)
        self._router.build()

        self._retrieval_grader = RetrievalGraderAgent(llm=self._llm)
        self._retrieval_grader.build()

        self._answer_grader = AnswerGraderAgent(llm=self._llm)
        self._answer_grader.build()

        self._hallucination_grader = HallucinationGraderAgent(llm=self._llm)
        self._hallucination_grader.build()

        self._generation = GenerationAgent(llm=self._llm)
        self._generation.build()

        self._reflection = ReflectionAgent(llm=self._llm)
        self._reflection.build()

        self._built = True
        return self

    @property
    def llm(self) -> ChatGroq:
        """Get the shared LLM instance."""
        if self._llm is None:
            raise RuntimeError("Call build() first")
        return self._llm

    @property
    def router(self) -> RouterAgent:
        if not hasattr(self, "_router") or self._router is None:
            raise RuntimeError("Call build() first")
        return self._router

    @property
    def retrieval_grader(self) -> RetrievalGraderAgent:
        if not hasattr(self, "_retrieval_grader") or self._retrieval_grader is None:
            raise RuntimeError("Call build() first")
        return self._retrieval_grader

    @property
    def answer_grader(self) -> AnswerGraderAgent:
        if not hasattr(self, "_answer_grader") or self._answer_grader is None:
            raise RuntimeError("Call build() first")
        return self._answer_grader

    @property
    def hallucination_grader(self) -> HallucinationGraderAgent:
        if not hasattr(self, "_hallucination_grader") or self._hallucination_grader is None:
            raise RuntimeError("Call build() first")
        return self._hallucination_grader

    @property
    def generation(self) -> GenerationAgent:
        if not hasattr(self, "_generation") or self._generation is None:
            raise RuntimeError("Call build() first")
        return self._generation

    @property
    def reflection(self) -> ReflectionAgent:
        if not hasattr(self, "_reflection") or self._reflection is None:
            raise RuntimeError("Call build() first")
        return self._reflection


def create_agent_builder(
    model: str = AgentBuilder.DEFAULT_MODEL,
    temperature: float = AgentBuilder.DEFAULT_TEMPERATURE,
) -> AgentBuilder:
    """Factory function to create an agent builder."""
    return AgentBuilder(model=model, temperature=temperature)
