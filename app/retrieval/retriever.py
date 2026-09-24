from typing import Any

from app.retrieval.embeddings import EmbeddingService
from app.retrieval.reranker import Reranker
from app.retrieval.vector_store import VectorStore


EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


class Retriever:
    """Retrieve candidates from Qdrant and rerank them."""

    def __init__(self) -> None:
        self.embedding_service = EmbeddingService(EMBEDDING_MODEL)
        self.vector_store = VectorStore()
        self.reranker = Reranker()

    def retrieve(
        self,
        query: str,
        *,
        retrieval_limit: int = 10,
        top_k: int = 5,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self.embedding_service.encode([query])[0]

        candidates = self.vector_store.search(
            query_embedding=query_embedding,
            limit=retrieval_limit,
            source=source,
        )

        return self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=top_k,
        )