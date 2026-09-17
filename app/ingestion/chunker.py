from typing import Any


def chunk_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[dict[str, Any]]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks = []

    for document in documents:
        text = document["text"].strip()
        metadata = document["metadata"]

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": {
                            **metadata,
                            "chunk_index": chunk_index,
                        },
                    }
                )
                chunk_index += 1

            if end >= len(text):
                break

            start = end - chunk_overlap

    return chunks