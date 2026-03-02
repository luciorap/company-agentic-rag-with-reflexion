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

### Models
- **LLM**: Groq (Llama 3.3 70B)
- **Embeddings**: HuggingFace (sentence-transformers/all-MiniLM-L6-v2)

### APIs
- **GROQ_API_KEY**: For ChatGroq
- **TAVILY_API_KEY**: For web search

### Technologies
- LangGraph (orchestration)
- LangChain (chains)
- Chroma (vectorstore)
- Tavily (web search)

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

### 1. Load documents (ingestion)

```bash
python ingestion.py
```

Loads documents from:
- Web: crunchbase.com, statista.com, apollo.io
- Internal: `./internal_docs/` (PDFs and DOCX)

### 2. Run agent

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
│   ├── chains/                 # LangChain chains
│   │   ├── answer_grader.py
│   │   ├── generation.py
│   │   ├── hallucination_grader.py
│   │   ├── reflection_chains.py
│   │   ├── retrieval_grader.py
│   │   └── router.py
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
