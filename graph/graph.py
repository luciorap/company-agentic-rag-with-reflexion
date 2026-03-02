from dotenv import load_dotenv
load_dotenv()

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH, REFLECT, EXECUTE_TOOLS
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.nodes.reflect import draft_node, revise_node, event_loop, MAX_ITERATIONS
from graph.nodes.tool_executor import execute_tools
from graph.state import GraphState
from graph.agent_builder import get_agent_builder


def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    builder = get_agent_builder()

    if not documents:
        print("---NO DOCUMENTS, SKIP HALLUCINATION CHECK---")
        print("---GRADE GENERATION vs QUESTION---")
        score = builder.answer_grader.run(question=question, generation=generation)
        if score.binary_score.lower() == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"

    score = builder.hallucination_grader.run(
        documents=documents, generation=generation
    )

    if score.binary_score.lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = builder.answer_grader.run(question=question, generation=generation)
        if score.binary_score.lower() == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    
    builder = get_agent_builder()
    source = builder.router.run(question=question)
    
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE
    
    return RETRIEVE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(REFLECT, draft_node)
workflow.add_node(EXECUTE_TOOLS, execute_tools)
workflow.add_node("revise", revise_node)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": REFLECT,
        "useful": END,
        "not useful": REFLECT,
    },
)

workflow.add_edge(REFLECT, EXECUTE_TOOLS)
workflow.add_edge(EXECUTE_TOOLS, "revise")
workflow.add_conditional_edges(
    "revise",
    event_loop,
    {
        "execute_tools": EXECUTE_TOOLS,
        "end": END,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()
