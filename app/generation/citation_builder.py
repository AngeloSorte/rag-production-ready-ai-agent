from typing import Any


def build_citations(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build reliable citations from retrieved document metadata."""

    citations = []

    for index, document in enumerate(documents, start=1):
        metadata = document["metadata"]

        citations.append(
            {
                "id": index,
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk": metadata["chunk_index"],
            }
        )

    return citations