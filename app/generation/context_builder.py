from typing import Any


def build_context(
    documents: list[dict[str, Any]],
) -> str:
    """Build the context that will be provided to the LLM."""

    context_parts = []

    for index, document in enumerate(documents, start=1):
        metadata = document["metadata"]

        source = metadata["source"]
        page = metadata["page"]
        chunk = metadata["chunk_index"]

        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk}\n"
            f"Content:\n{document['text']}"
        )

    return "\n\n".join(context_parts)