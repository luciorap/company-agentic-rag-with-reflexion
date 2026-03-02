import datetime
from typing import TYPE_CHECKING, Any, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

if TYPE_CHECKING:
    from langchain_groq import ChatGroq

from graph.schemas import AnswerQuestion, ReviseAnswer


class ReflectionAgent:
    """Agent for reflection and iterative improvement of answers."""

    def __init__(self, llm: "ChatGroq"):
        """Initialize with a shared LLM instance.
        
        Args:
            llm: Shared ChatGroq instance.
        """
        self.llm = llm
        self._first_responder = None
        self._revisor = None

    def build(self):
        """Build the reflection chains (first responder and revisor)."""
        actor_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are expert researcher.
Current time: {time}

1. {first_instruction}
2. Reflect and critique your answer. Be severe to maximize improvement.
3. Recommend search queries to research information and improve your answer.""",
                ),
                MessagesPlaceholder(variable_name="messages"),
                ("system", "Answer the user's question above using the required format."),
            ]
        ).partial(
            time=lambda: datetime.datetime.now().isoformat(),
        )

        first_responder_prompt_template = actor_prompt_template.partial(
            first_instruction="Provide a detailed ~250 word answer with citations."
        )

        self._first_responder = first_responder_prompt_template | self.llm.bind_tools(
            tools=[AnswerQuestion], tool_choice="AnswerQuestion"
        )

        revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

        self._revisor = actor_prompt_template.partial(
            first_instruction=revise_instructions
        ) | self.llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

        return self

    def draft(self, question: str, messages: List[BaseMessage] = None) -> Any:
        """Run the first responder to draft an initial answer."""
        if self._first_responder is None:
            self.build()
        
        if messages is None:
            messages = []
        
        return self._first_responder.invoke({"messages": messages})

    def revise(self, messages: List[BaseMessage]) -> Any:
        """Run the revisor to improve the answer based on tool results."""
        if self._revisor is None:
            self.build()
        
        return self._revisor.invoke({"messages": messages})


def create_reflection_agent(llm: "ChatGroq") -> ReflectionAgent:
    """Factory function to create a reflection agent with shared LLM."""
    return ReflectionAgent(llm=llm)
