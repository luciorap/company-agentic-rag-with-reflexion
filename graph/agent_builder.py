from graph.chains import AgentBuilder

_agent_builder = None


def get_agent_builder() -> AgentBuilder:
    """Get or create the global agent builder."""
    global _agent_builder
    if _agent_builder is None:
        _agent_builder = AgentBuilder().build()
    return _agent_builder


def reset_agent_builder():
    """Reset the agent builder (useful for testing)."""
    global _agent_builder
    _agent_builder = None
