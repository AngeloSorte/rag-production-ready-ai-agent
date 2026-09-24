from typing import Protocol

from groq import Groq

from app.core.config import settings


class LLM(Protocol):
    """Interface for a Large Language Model."""

    def generate(self, prompt: str) -> str:
        ...


class GroqLLM:
    """LLM implementation using Groq API."""

    def __init__(
        self,
        model: str = "openai/gpt-oss-20b",
    ) -> None:
        if not settings.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set")

        self.client = Groq(
            api_key=settings.groq_api_key
        )
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
        )

        return response.choices[0].message.content or ""