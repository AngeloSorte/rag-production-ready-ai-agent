from typing import Any


SYSTEM_PROMPT = """You are a helpful AI assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information to answer the question,
say that the available documents do not contain enough information.

Do not invent facts.

Do not generate source citations, page numbers, or references in your answer.
The application will provide citations separately based on the retrieved
document metadata.

Answer clearly and directly.
"""


def build_prompt(
    query: str,
    documents: list[dict[str, Any]],
) -> str:
    context_parts = []

    for index, document in enumerate(documents, start=1):
        metadata = document["metadata"]

        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {metadata['source']}\n"
            f"Page: {metadata['page']}\n"
            f"Chunk: {metadata['chunk_index']}\n"
            f"Content:\n{document['text']}"
        )

    context = "\n\n".join(context_parts)

    return f"""{SYSTEM_PROMPT}

CONTEXT:
{context}

USER QUESTION:
{query}

ANSWER:
"""

