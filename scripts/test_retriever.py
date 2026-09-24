from app.retrieval.retriever import Retriever


def main() -> None:
    query = "Che cosa meditiamo nel Santo Rosario?"

    print(f"Query: {query}")
    print("\nRunning complete retrieval pipeline...\n")

    retriever = Retriever()

    results = retriever.retrieve(
        query,
        retrieval_limit=10,
        top_k=5,
    )

    for index, result in enumerate(results, start=1):
        print(f"--- Result {index} ---")
        print(f"Rerank score: {result['rerank_score']:.4f}")
        print(f"Vector score: {result['score']:.4f}")
        print(f"Page: {result['metadata']['page']}")
        print(f"Chunk: {result['metadata']['chunk_index']}")
        print(f"Text: {result['text'][:300]}")
        print()


if __name__ == "__main__":
    main()