# Company Agentic RAG with Reflexion

Implementation of Reflective RAG, Self-RAG & Adaptive RAG tailored towards developers and production-oriented applications for learning LangGraph.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Company Agentic RAG                              │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────────┐
                    │      route_question          │
                    │   (vectorstore vs websearch) │
                    └──────────────┬───────────────┘
                                   │
            ┌──────────────────────┴──────────────────────┐
            │                                                 │
            ▼                                                 ▼
    ┌───────────────┐                               ┌───────────────┐
    │   RETRIEVE   │                               │  WEB SEARCH   │
    │  (Chroma)    │                               │  (Tavily)     │
    └───────┬───────┘                               └───────┬───────┘
            │                                                 │
            ▼                                                 ▼
    ┌───────────────┐                               ┌───────────────┐
    │GRADE_DOCUMENTS│                               │   GENERATE   │
    │               │                               │               │
    └───────┬───────┘                               └───────┬───────┘
            │                                                 │
            ▼                                                 ▼
    ┌───────────────┐                               ┌───────────────┐
    │ decide_to_    │                               │   REFLECT     │
    │ generate      │                               │ (draft_node)  │
    └───────┬───────┘                               └───────┬───────┘
            │                                                 │
            ▼                                                 ▼
    ┌───────────────┐                               ┌───────────────┐
    │   GENERATE    │                               │  EXECUTE_    │
    │               │                               │    TOOLS     │
    └───────┬───────┘                               └───────┬───────┘
            │                                                 │
            ▼                                                 ▼
    ┌───────────────┐                               ┌───────────────┐
    │   HALLU-      │                               │    REVISE     │
    │   CINATION    │                               │               │
    │   CHECK       │                               └───────┬───────┘
    └───────┬───────┘                                       │
            │                                               ▼
            │                                     ┌───────────────┐
            │                                     │  event_loop   │
            └──────────────┬──────────────────────┤ (max 2 iter)  │
                           │                      └───────┬───────┘
                           ▼                              │
                    ┌─────────────┐                       │
                    │  REFLECT    │◄──────────────────────┘
                    │  (retry)    │
                    └─────────────┘
```

## Components

### AgentBuilder (Main Entry Point)

All agents are configured once using the `AgentBuilder` class, which creates a single LLM instance shared across all agents.

```python
from graph.chains import AgentBuilder

# Configure once - model and temperature
builder = AgentBuilder(
    model="llama-3.3-70b-versatile",
    temperature=0
).build()

# Access all agents
builder.llm                    # Shared LLM instance
builder.router                 # Route questions
builder.retrieval_grader       # Grade document relevance
builder.answer_grader         # Check if answer addresses question
builder.hallucination_grader  # Verify grounded in facts
builder.generation            # Generate answers
builder.reflection            # Draft & revise with reflection
```

### Agent Classes

| Agent | Purpose |
|-------|---------|
| `RouterAgent` | Routes questions to vectorstore or web search |
| `RetrievalGraderAgent` | Grades document relevance to question |
| `AnswerGraderAgent` | Checks if answer addresses the question |
| `HallucinationGraderAgent` | Verifies generation is grounded in facts |
| `GenerationAgent` | Generates answers from context |
| `ReflectionAgent` | Drafts and revises answers with iterative reflection |

### Technologies
- **LLM**: Groq (Llama 3.3 70B)
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)
- **Vectorstore**: Chroma
- **Web Search**: Tavily
- **Orchestration**: LangGraph
- **Chains**: LangChain

## Environment Variables

```bash
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional
LANGCHAIN_TRACING_V2=true                       # Optional
LANGCHAIN_PROJECT=company-agentic-rag           # Optional
USER_AGENT=company-agentic-rag/1.0              # Optional
```

## Installation

```bash
# Clone repository
git clone https://github.com/luciorap/company-agentic-rag-with-reflexion.git
cd company-agentic-rag-with-reflexion

# Install dependencies
pip install -r requirements.txt

# Or with Poetry
poetry install
```

## Usage

### 1. Load Documents (Ingestion)

```bash
python ingestion.py
```

Loads documents from:
- Web: crunchbase.com, statista.com, apollo.io
- Internal: `./internal_docs/` (PDFs and DOCX)

### 2. Run Agent

```bash
python main.py
```

### Agent Flow

1. **Route Question**: Decide whether to use RAG or Web Search
2. **Retrieve**: Get relevant documents from vectorstore
3. **Grade Documents**: Evaluate document relevance
4. **Generate**: Generate answer based on documents
5. **Check Hallucinations**: Verify answer is grounded
6. **Reflect** (if not grounded): Rewrite with max 2 iterations

## Project Structure

```
.
├── graph/                      # LangGraph graph
│   ├── chains/                 # Agent classes
│   │   ├── builder.py          # AgentBuilder (main entry)
│   │   ├── router.py           # RouterAgent
│   │   ├── generation.py       # GenerationAgent
│   │   ├── answer_grader.py    # AnswerGraderAgent
│   │   ├── hallucination_grader.py
│   │   ├── retrieval_grader.py
│   │   └── reflection_chains.py
│   ├── nodes/                  # Graph nodes
│   │   ├── generate.py
│   │   ├── grade_documents.py
│   │   ├── reflect.py
│   │   ├── retrieve.py
│   │   ├── tool_executor.py
│   │   └── web_search.py
│   ├── graph.py                # Graph definition
│   ├── state.py                # Graph state
│   └── consts.py               # Constants
├── ingestion.py                # Document loading
├── main.py                     # Entry point
└── .env                        # Environment variables
```

## Tests

```bash
pytest . -s -v
```

## Customization

### Change Model

```python
from graph.chains import AgentBuilder

builder = AgentBuilder(
    model="llama-3.1-70b-versatile",  # Change model
    temperature=0.5                    # Adjust creativity
).build()
```

### Use Custom LLM

```python
from langchain_groq import ChatGroq
from graph.chains import AgentBuilder

# Create your own LLM instance
custom_llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)

# Pass to builder (requires modifying AgentBuilder to accept llm parameter)
```
