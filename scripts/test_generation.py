from app.generation.generator import Generator
from app.generation.llm import GroqLLM
from app.retrieval.retriever import Retriever


def main() -> None:
    query = "Che cosa meditiamo nel Santo Rosario?"

    print(f"Query: {query}")
    print("\n1. Retrieving documents...\n")

    retriever = Retriever()

    documents = retriever.retrieve(
        query,
        retrieval_limit=10,
        top_k=5,
    )

    print(f"Documents retrieved: {len(documents)}")

    print("\n2. Generating answer with Groq...\n")

    generator = Generator(
        llm=GroqLLM(),
    )

    result = generator.generate(
        query=query,
        documents=documents,
    )

    print("=== ANSWER ===")
    print(result["answer"])

    print("\n=== CITATIONS ===")
    for citation in result["citations"]:
        print(citation)


if __name__ == "__main__":
    main()