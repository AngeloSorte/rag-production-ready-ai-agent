\# RAG Production-Ready AI Agent



A production-oriented Retrieval-Augmented Generation (RAG) system built with Python, FastAPI, Qdrant, Sentence Transformers, and Groq.



The project implements an end-to-end RAG pipeline that ingests documents, creates semantic embeddings, retrieves relevant information from a vector database, reranks the retrieved candidates, generates grounded answers with an LLM, and returns structured source citations.



> \*\*Status:\*\* Work in Progress — core RAG pipeline implemented and retrieval evaluation started.



\---



\## Architecture



```text

User Question

&#x20;     │

&#x20;     ▼

Query Embedding

&#x20;     │

&#x20;     ▼

Qdrant Vector Search

&#x20;     │

&#x20;     ▼

Top-N Candidates

&#x20;     │

&#x20;     ▼

Cross-Encoder Reranking

&#x20;     │

&#x20;     ▼

Top-K Relevant Documents

&#x20;     │

&#x20;     ├──────────────► Structured Citations

&#x20;     │

&#x20;     ▼

Context Construction

&#x20;     │

&#x20;     ▼

Prompt Construction

&#x20;     │

&#x20;     ▼

Groq LLM

&#x20;     │

&#x20;     ▼

Grounded Answer

```



The system separates retrieval, generation, and infrastructure concerns so that individual components can be replaced or evaluated independently.



\---



\## Key Features



\* PDF document ingestion with PyMuPDF

\* Document chunking with configurable chunk size and overlap

\* Multilingual semantic embeddings

\* Local persistent Qdrant vector database

\* Metadata-aware retrieval

\* Semantic similarity search

\* Cross-encoder reranking

\* LLM abstraction through a provider-independent interface

\* Groq LLM integration

\* Context-aware prompt construction

\* Structured source citations

\* Retrieval evaluation with Hit@K

\* Environment-based configuration

\* FastAPI application foundation

\* Automated testing structure

\* Git-based version control



\---



\## Technology Stack



| Component                  | Technology                              |

| -------------------------- | --------------------------------------- |

| Language                   | Python 3.12                             |

| API                        | FastAPI                                 |

| Validation / Configuration | Pydantic + pydantic-settings            |

| Document Parsing           | PyMuPDF                                 |

| Embeddings                 | Sentence Transformers                   |

| Embedding Model            | `paraphrase-multilingual-MiniLM-L12-v2` |

| Vector Database            | Qdrant                                  |

| Distance Metric            | Cosine similarity                       |

| Reranker                   | `cross-encoder/ms-marco-MiniLM-L6-v2`   |

| LLM Provider               | Groq                                    |

| LLM                        | `openai/gpt-oss-20b`                    |

| Testing                    | pytest                                  |

| Linting                    | Ruff                                    |

| Version Control            | Git                                     |



\---



\## Project Structure



```text

rag-production-ready-ai-agent/

│

├── app/

│   ├── api/

│   ├── core/

│   │   └── config.py

│   ├── evaluation/

│   ├── generation/

│   │   ├── citation\_builder.py

│   │   ├── context\_builder.py

│   │   ├── generator.py

│   │   ├── llm.py

│   │   └── prompt\_builder.py

│   ├── ingestion/

│   │   ├── chunker.py

│   │   ├── embedding\_service.py

│   │   └── pdf\_loader.py

│   ├── models/

│   └── retrieval/

│       ├── embeddings.py

│       ├── reranker.py

│       ├── retriever.py

│       └── vector\_store.py

│

├── data/

│   ├── raw/

│   ├── processed/

│   └── qdrant/

│

├── scripts/

│   ├── index\_documents.py

│   ├── test\_context\_builder.py

│   ├── test\_embeddings.py

│   ├── test\_generation.py

│   ├── test\_prompt\_builder.py

│   ├── test\_reranker.py

│   ├── test\_retrieval.py

│   └── test\_retriever.py

│

├── tests/

│   ├── evaluation/

│   │   ├── test\_cases.json

│   │   └── test\_retrieval.py

│   ├── integration/

│   └── unit/

│

├── .env.example

├── .gitignore

├── pyproject.toml

└── README.md

```



\---



\## How the RAG Pipeline Works



\### 1. Document ingestion



PDF documents are parsed and converted into text while preserving metadata such as:



\* source document

\* page number



\### 2. Chunking



Documents are divided into smaller pieces before embedding.



Each chunk retains its original metadata and receives a chunk index.



This allows the system to retrieve information while still knowing exactly where the information came from.



\### 3. Embeddings



Each chunk is transformed into a numerical vector using:



```text

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

```



The model produces 384-dimensional embeddings and supports multiple languages.



\### 4. Vector search



Embeddings are stored locally in Qdrant.



When a user asks a question:



```text

Question

&#x20;  ↓

Question embedding

&#x20;  ↓

Qdrant similarity search

&#x20;  ↓

Top-N candidates

```



The system can also filter results using document metadata.



\### 5. Reranking



The initial vector search provides candidate documents.



A cross-encoder then evaluates the relationship between the question and each candidate and produces a reranking score.



This creates a two-stage retrieval architecture:



```text

Fast candidate retrieval

&#x20;       ↓

More precise reranking

```



\### 6. Context construction



The highest-ranked documents are assembled into a context that is provided to the language model.



The context contains the document text and source metadata.



\### 7. Generation



The LLM receives:



\* system instructions

\* retrieved context

\* user question



The prompt explicitly instructs the model to answer using the supplied context and avoid inventing unsupported information.



\### 8. Citations



Citations are generated independently from the LLM response using the metadata of the retrieved documents.



This prevents the language model from inventing page numbers or source references.



Example:



```json

{

&#x20; "id": 1,

&#x20; "source": "document.pdf",

&#x20; "page": 2,

&#x20; "chunk": 0

}

```



\---



\## Evaluation



The project includes an initial retrieval evaluation suite.



The evaluation currently contains five representative questions and checks whether an expected source is found among the top five retrieved results.



\### Current result



```text

=== SUMMARY ===

Passed: 5/5

Hit@5: 100.00%

```



This means that, for the current five-question evaluation set, the expected source was found within the top five retrieved documents for every question.



\*\*Important:\*\* Hit@5 is a retrieval metric. It does not mean that the complete RAG system has 100% answer accuracy.



Future evaluation will include additional metrics for retrieval quality and generated-answer grounding.



Run the current retrieval evaluation with:



```bash

python -m tests.evaluation.test\_retrieval

```



\---



\## Configuration



Create a `.env` file based on `.env.example`.



Example:



```env

APP\_NAME=RAG Production-Ready AI Agent

APP\_VERSION=0.1.0

ENVIRONMENT=development

GROQ\_API\_KEY=your\_api\_key\_here

```



Secrets are intentionally excluded from version control.



The real `.env` file is ignored by Git.



\---



\## Running the Project



Create and activate a virtual environment:



```bash

python -m venv .venv

```



Windows:



```powershell

.venv\\Scripts\\Activate.ps1

```



Install the project dependencies:



```bash

pip install -e .

```



Run the FastAPI application:



```bash

uvicorn app.main:app --reload

```



The health endpoint is available at:



```text

http://127.0.0.1:8000/health

```



\---



\## Indexing Documents



Place source PDF documents inside:



```text

data/raw/

```



Then run:



```bash

python -m scripts.index\_documents

```



The pipeline will:



1\. load the PDF

2\. extract document text

3\. create chunks

4\. generate embeddings

5\. create the Qdrant collection if necessary

6\. store the vectors and metadata



\---



\## Testing Individual Components



Examples:



```bash

python -m scripts.test\_embeddings

python -m scripts.test\_retrieval

python -m scripts.test\_reranker

python -m scripts.test\_retriever

python -m scripts.test\_generation

```



The project also contains a dedicated evaluation suite under:



```text

tests/evaluation/

```



\---



\## Design Principles



\### Separation of concerns



Retrieval and generation are deliberately separated.



The ma



