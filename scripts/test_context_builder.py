from app.generation.context_builder import build_context
from app.retrieval.retriever import Retriever


def main() -> None:
    query = "Che cosa meditiamo nel Santo Rosario?"

    retriever = Retriever()

    results = retriever.retrieve(
        query,
        retrieval_limit=10,
        top_k=5,
    )

    context = build_context(results)

    print("=== GENERATED CONTEXT ===\n")
    print(context)


if __name__ == "__main__":
    main()