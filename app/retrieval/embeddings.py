from typing import Any

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Generate semantic embeddings for text."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
    ) -> list[list[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return embeddings.tolist()

    def embed_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        texts = [document["text"] for document in documents]
        embeddings = self.encode(texts)

        result = []

        for document, embedding in zip(documents, embeddings):
            result.append(
                {
                    **document,
                    "embedding": embedding,
                }
            )

        return result