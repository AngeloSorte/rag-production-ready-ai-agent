from typing import Any

from app.generation.citation_builder import build_citations
from app.generation.llm import LLM
from app.generation.prompt_builder import build_prompt


class Generator:
    """Generate answers using retrieved documents and an LLM."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def generate(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = build_prompt(
            query=query,
            documents=documents,
        )

        answer = self.llm.generate(prompt)

        citations = build_citations(documents)

        return {
            "answer": answer,
            "citations": citations,
        }