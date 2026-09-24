from app.retrieval.embeddings import EmbeddingService
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore


EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def main() -> None:
    query = "Che cosa meditiamo nel Santo Rosario?"

    print(f"Query: {query}")
    print("\n1. Retrieving candidates...\n")

    embedding_service = EmbeddingService(EMBEDDING_MODEL)
    vector_store = VectorStore()

    query_embedding = embedding_service.encode([query])[0]

    candidates = vector_store.search(
        query_embedding=query_embedding,
        limit=10,
    )

    print(f"Candidates retrieved: {len(candidates)}")

    print("\n2. Loading reranker...\n")

    reranker = Reranker()

    print("\n3. Reranking candidates...\n")

    results = reranker.rerank(
        query=query,
        documents=candidates,
        top_k=5,
    )

    for index, result in enumerate(results, start=1):
        print(f"--- Reranked Result {index} ---")
        print(f"Rerank score: {result['rerank_score']:.4f}")
        print(f"Vector score: {result['score']:.4f}")
        print(f"Page: {result['metadata']['page']}")
        print(f"Chunk: {result['metadata']['chunk_index']}")
        print(f"Text: {result['text'][:300]}")
        print()


if __name__ == "__main__":
    main()