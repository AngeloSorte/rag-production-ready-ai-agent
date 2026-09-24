from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    embedding_service = EmbeddingService(MODEL_NAME)
    vector_store = VectorStore()

    query = "Che cosa meditiamo nel Santo Rosario?"

    print(f"Query: {query}")
    print("\nSearching...\n")

    query_embedding = embedding_service.encode([query])[0]

    results = vector_store.search(
        query_embedding=query_embedding,
        limit=5,
    )

    for index, result in enumerate(results, start=1):
        print(f"--- Result {index} ---")
        print(f"Score: {result['score']:.4f}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Page: {result['metadata']['page']}")
        print(f"Chunk: {result['metadata']['chunk_index']}")
        print(f"Text: {result['text'][:300]}")
        print()


if __name__ == "__main__":
    main()