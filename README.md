# RAG Production-Ready AI Agent

A production-oriented Retrieval-Augmented Generation (RAG) system built with Python, FastAPI, Qdrant, Sentence Transformers, and Groq.

The project implements an end-to-end RAG pipeline that ingests documents, creates semantic embeddings, retrieves relevant information from a vector database, reranks candidate documents, generates grounded answers with an LLM, and returns structured source citations.

The architecture is designed around separation of concerns, testability, evaluation, and replaceable AI components.

---

## Architecture

```text
                         USER QUESTION
                              |
                              v
                    +--------------------+
                    | Query Embedding    |
                    +--------------------+
                              |
                              v
                    +--------------------+
                    | Qdrant Vector      |
                    | Similarity Search   |
                    +--------------------+
                              |
                              v
                       Top-N Candidates
                              |
                              v
                    +--------------------+
                    | Italian            |
                    | Cross-Encoder      |
                    | Reranker           |
                    +--------------------+
                              |
                              v
                       Top-K Documents
                              |
                    +---------+---------+
                    |                   |
                    v                   v
             Context Builder      Citation Builder
                    |
                    v
             Prompt Construction
                    |
                    v
                  Groq LLM
                    |
                    v
             Grounded Answer

The system separates retrieval, generation, API, configuration, and evaluation concerns so individual components can be tested or replaced independently.

Key Features
PDF document ingestion with PyMuPDF
Configurable document chunking
Multilingual semantic embeddings
384-dimensional embedding vectors
Local persistent Qdrant vector database
Cosine similarity search
Metadata-aware retrieval
Two-stage retrieval architecture
Italian-specific cross-encoder reranking
Provider-independent LLM interface
Groq LLM integration
Grounded prompt construction
Structured source citations generated independently from the LLM
Retrieval evaluation with Hit@5
Generation evaluation cases
FastAPI REST API
Request validation with Pydantic
Environment-based configuration
Retry handling for transient LLM failures
Unit/integration/evaluation test structure
Ruff linting
Docker support
Git-based version control
Technology Stack
Component	Technology
Language	Python 3.12
API	FastAPI
Validation / Configuration	Pydantic + pydantic-settings
Document Parsing	PyMuPDF
Embeddings	Sentence Transformers
Embedding Model	sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Embedding Dimension	384
Vector Database	Qdrant
Distance Metric	Cosine similarity
Reranker	vincolle/reranker-bert-italian-uncased-mmarco-mnrl
LLM Provider	Groq
LLM	openai/gpt-oss-20b
Testing	pytest
Linting	Ruff
Containerization	Docker
Version Control	Git
Project Structure
rag-production-ready-ai-agent/
│
├── app/
│   ├── api/
│   │   └── routes.py
│   │
│   ├── core/
│   │   └── config.py
│   │
│   ├── generation/
│   │   ├── citation_builder.py
│   │   ├── context_builder.py
│   │   ├── generator.py
│   │   ├── llm.py
│   │   └── prompt_builder.py
│   │
│   ├── ingestion/
│   │   ├── chunker.py
│   │   └── pdf_loader.py
│   │
│   ├── models/
│   │
│   └── retrieval/
│       ├── embeddings.py
│       ├── reranker.py
│       ├── retriever.py
│       └── vector_store.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── qdrant/
│
├── scripts/
│   ├── index_documents.py
│   ├── test_context_builder.py
│   ├── test_embeddings.py
│   ├── test_generation.py
│   ├── test_prompt_builder.py
│   ├── test_reranker.py
│   ├── test_retrieval.py
│   └── test_retriever.py
│
├── tests/
│   ├── evaluation/
│   │   ├── test_cases.json
│   │   ├── test_generation_cases.json
│   │   ├── test_generation.py
│   │   ├── test_grounding.py
│   │   └── test_retrieval.py
│   │
│   ├── integration/
│   │   └── test_api.py
│   │
│   └── unit/
│
├── .env.example
├── .gitignore
├── Dockerfile
├── pyproject.toml
└── README.md
How the RAG Pipeline Works
1. Document ingestion

PDF documents are loaded with PyMuPDF.

The loader extracts the text while preserving metadata such as:

source document
page number

This metadata is retained throughout the pipeline so retrieved information can later be traced back to its origin.

2. Chunking

Documents are divided into smaller text chunks before embedding.

The current implementation uses configurable:

chunk size
chunk overlap

Each chunk preserves the original document metadata and receives a chunk index.

This allows the system to retrieve smaller relevant portions of a document instead of passing an entire document to the LLM.

3. Embeddings

Each chunk is transformed into a numerical vector using:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

The model produces 384-dimensional embeddings.

The same embedding model is used for both documents and user queries, allowing semantic similarity search.

Conceptually:

Text
  |
  v
Embedding Model
  |
  v
384-dimensional vector
4. Vector Search

The embeddings are stored in a persistent local Qdrant collection.

When a user asks a question:

User Question
      |
      v
Question Embedding
      |
      v
Qdrant Similarity Search
      |
      v
Top-N Candidates

The current retrieval pipeline initially retrieves up to 10 candidates.

Qdrant uses cosine similarity to identify semantically related vectors.

The vector store also supports filtering using document metadata such as the source document.

5. Reranking

Vector similarity provides fast candidate retrieval, but the initial ranking is not necessarily the final ranking.

The system therefore performs a second retrieval stage using an Italian-specific cross-encoder:

vincolle/reranker-bert-italian-uncased-mmarco-mnrl

The reranker evaluates the relationship between:

(query, candidate document)

and assigns a relevance score.

The current pipeline therefore follows:

Fast semantic retrieval
        |
        v
10 candidates
        |
        v
Cross-encoder reranking
        |
        v
Top 5 documents

This two-stage architecture separates high-recall candidate retrieval from more precise relevance ranking.

6. Context Construction

The highest-ranked documents are assembled into a context for the language model.

Each context item contains:

document source
page number
chunk number
retrieved text

Example:

[Source 1]
Document: document.pdf
Page: 2
Chunk: 0
Content:
...

This gives the LLM both the content and the provenance information associated with the retrieved text.

7. Prompt Construction

The prompt contains three main elements:

System instructions
        +
Retrieved context
        +
User question

The system prompt explicitly instructs the LLM to:

use only the supplied context
avoid outside knowledge
avoid unsupported inference
avoid inventing facts
clearly state when the context is insufficient

The goal is to reduce unsupported generation and keep the answer grounded in retrieved evidence.

8. LLM Generation

The current implementation uses Groq as the LLM provider with:

openai/gpt-oss-20b

The LLM receives the constructed prompt and produces the final answer.

The LLM is accessed through an abstraction:

LLM interface
     |
     +---- Groq implementation

This keeps the generation layer replaceable if another provider or local model is introduced later.

The Groq implementation also contains retry handling for connection/API failures and empty responses.

9. Structured Citations

Citations are generated independently from the LLM response.

The application builds citations directly from retrieved document metadata.

Example:

{
  "id": 1,
  "source": "document.pdf",
  "page": 2,
  "chunk": 0
}

This is intentional.

The LLM does not have to invent or reproduce page numbers and source references.

The application already knows where each retrieved chunk came from.

The API therefore returns:

{
  "answer": "...",
  "citations": [
    {
      "id": 1,
      "source": "document.pdf",
      "page": 2,
      "chunk": 0
    }
  ]
}
API

The application exposes a FastAPI REST API.

Health Check
GET /health

Example response:

{
  "status": "ok",
  "environment": "development"
}
Ask a Question
POST /api/ask

Request:

{
  "question": "Che cosa meditiamo nel Santo Rosario?"
}

Response:

{
  "answer": "Generated answer...",
  "citations": [
    {
      "id": 1,
      "source": "document.pdf",
      "page": 1,
      "chunk": 0
    }
  ]
}
Interactive API Documentation

When the application is running:

http://127.0.0.1:8000/docs

FastAPI provides interactive Swagger documentation for the available endpoints.

Evaluation

The project includes an initial retrieval evaluation suite.

The current retrieval dataset contains:

4 answerable questions
1 explicitly unanswerable question

Answerable questions are evaluated using Hit@5.

Current Retrieval Result
=== SUMMARY ===
Answerable cases: 4/4
Hit@5: 100.00%

This means that, in the current evaluation dataset, the expected source was found within the top five retrieved documents for all four answerable questions.

The unanswerable question is intentionally excluded from Hit@5 because it has no expected source.

Important

The current 100% Hit@5 result does not mean that the complete RAG system has 100% answer accuracy.

It is a retrieval metric measured on a small evaluation dataset.

It does not by itself measure:

factual correctness of the generated answer
completeness of the answer
hallucination rate
citation correctness
production-scale robustness

The project also contains generation and grounding evaluation components that are being developed alongside retrieval evaluation.

Run retrieval evaluation with:

python -m tests.evaluation.test_retrieval
Testing

Run the complete pytest suite with:

python -m pytest -v

Integration tests include:

health endpoint
valid /api/ask requests
invalid empty questions
HTTP response validation

Individual component checks are also available:

python -m scripts.test_embeddings
python -m scripts.test_retrieval
python -m scripts.test_reranker
python -m scripts.test_retriever
python -m scripts.test_generation
Configuration

Create a .env file based on .env.example.

Example:

APP_NAME=RAG Production-Ready AI Agent
APP_VERSION=0.1.0
ENVIRONMENT=development
GROQ_API_KEY=your_api_key_here

Secrets are intentionally excluded from version control.

The real .env file is ignored by Git.

Running Locally

Create a virtual environment:

python -m venv .venv

On Windows:

.venv\Scripts\Activate.ps1

Install the project:

pip install -e .

Run the API:

uvicorn app.main:app --reload

The API will be available at:

http://127.0.0.1:8000

Swagger documentation:

http://127.0.0.1:8000/docs

Health endpoint:

http://127.0.0.1:8000/health
Indexing Documents

Place source PDF files inside:

data/raw/

Then run:

python -m scripts.index_documents

The indexing pipeline:

loads the PDF
extracts text
creates chunks
generates embeddings
creates the Qdrant collection if necessary
stores vectors and metadata

The local Qdrant database is stored under:

data/qdrant/
Docker

The project includes a Docker image for reproducible execution.

Build the image:

docker build -t rag-production-ready-ai-agent .

Run the container:

docker run --rm -p 8000:8000 --env-file .env rag-production-ready-ai-agent

The Docker image includes:

Python 3.12
application dependencies
CPU-only PyTorch
application source code
the local Qdrant data used by the project
the embedding model
the Italian reranker model

The AI model files are downloaded during image construction rather than during normal container startup.

Model initialization still occurs when the application process starts.

Once the container is running:

http://localhost:8000/health

Swagger:

http://localhost:8000/docs

The Docker setup is intended to make the application environment reproducible independently of the developer's local Python environment.

Design Principles
Separation of concerns

Retrieval, generation, API, configuration, and evaluation are implemented as separate components.

This makes individual components easier to test and replace.

Grounded generation

The LLM is explicitly instructed to use only the retrieved context.

Independent citations

Citations are generated by the application from document metadata rather than being invented by the LLM.

Evaluation-driven development

Retrieval quality is measured with an explicit evaluation dataset instead of relying only on manual inspection.

Replaceable AI components

The LLM is accessed through an abstraction so the provider can be replaced without redesigning the complete generation layer.

Reproducibility

Environment configuration, dependency definitions, Git version control, and Docker are used to make execution more reproducible.

Current Limitations

This is a portfolio-grade engineering project and not yet a complete production deployment.

Current limitations include:

small evaluation datasets
character-based chunking rather than advanced semantic chunking
local Qdrant persistence
local model execution for embeddings and reranking
external Groq dependency for generation
no authentication or authorization layer
no production monitoring/observability stack
no distributed deployment
dependency versions are not fully locked
Docker startup can still involve significant model initialization time

These limitations are intentional areas for future engineering work rather than hidden assumptions.

Roadmap

Potential next improvements include:

stronger automated retrieval evaluation
more robust answer-grounding evaluation
citation-level evaluation
semantic or structure-aware chunking
hybrid retrieval
query rewriting
configurable retrieval parameters
asynchronous API execution
authentication and authorization
structured logging and observability
dependency locking
production deployment
CI/CD pipeline
larger evaluation datasets
latency and performance benchmarks
Portfolio Summary

This project demonstrates an end-to-end implementation of a modern RAG system rather than only an LLM API call.

It covers:

Document Ingestion
        ↓
Chunking
        ↓
Embeddings
        ↓
Vector Database
        ↓
Semantic Retrieval
        ↓
Cross-Encoder Reranking
        ↓
Context Construction
        ↓
Grounded Prompting
        ↓
LLM Generation
        ↓
Structured Citations
        ↓
Evaluation
        ↓
FastAPI
        ↓
Docker

The project is intended to demonstrate practical AI engineering skills including retrieval architecture, vector search, reranking, LLM integration, API development, evaluation, testing, configuration management, and containerization.
