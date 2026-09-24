from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue
from qdrant_client.models import PointStruct, VectorParams


COLLECTION_NAME = "documents"
VECTOR_SIZE = 384


class VectorStore:
    def __init__(self, path: str = "data/qdrant") -> None:
        self.client = QdrantClient(path=path)

    def create_collection(self) -> None:
        collections = self.client.get_collections().collections

        if COLLECTION_NAME not in [collection.name for collection in collections]:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def add_documents(
        self,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunks and embeddings must have the same length")

        points = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):
            points.append(
                PointStruct(
                    id=index,
                    vector=embedding,
                    payload={
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    },
                )
            )

        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        query_filter = None

        if source:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=source),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "score": result.score,
                "text": result.payload["text"],
                "metadata": result.payload["metadata"],
            }
            for result in results.points
        ]