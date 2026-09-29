from typing import Any

from sentence_transformers import CrossEncoder


MODEL_NAME = "vincolle/reranker-bert-italian-uncased-mmarco-mnrl"

class Reranker:
    """Rerank retrieved documents using a lightweight cross-encoder."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        pairs = [
            [query, document["text"]]
            for document in documents
        ]

        scores = self.model.predict(pairs)

        scored_documents = []

        for document, score in zip(documents, scores):
            scored_documents.append(
                {
                    **document,
                    "rerank_score": float(score),
                }
            )

        scored_documents.sort(
            key=lambda document: document["rerank_score"],
            reverse=True,
        )

        return scored_documents[:top_k]