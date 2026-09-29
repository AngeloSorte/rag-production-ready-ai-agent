import logging
import time
from typing import Protocol

from groq import APIConnectionError
from groq import APIStatusError
from groq import Groq

from app.core.config import settings


logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0


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
        for attempt in range(1, MAX_RETRIES + 2):
            logger.info(
                "Sending generation request to Groq "
                "model=%s attempt=%s",
                self.model,
                attempt,
            )

            try:
                response = (
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        temperature=0,
                    )
                )

            except APIConnectionError as exc:
                logger.warning(
                    "Groq connection error on attempt %s: %s",
                    attempt,
                    exc,
                )

                if attempt > MAX_RETRIES:
                    raise RuntimeError(
                        "The LLM service could not be reached."
                    ) from exc

                time.sleep(RETRY_DELAY_SECONDS)
                continue

            except APIStatusError as exc:
                logger.warning(
                    "Groq API error on attempt %s: "
                    "status_code=%s",
                    attempt,
                    exc.status_code,
                )

                if attempt > MAX_RETRIES:
                    raise RuntimeError(
                        "The LLM service returned an error."
                    ) from exc

                time.sleep(RETRY_DELAY_SECONDS)
                continue

            answer = response.choices[0].message.content

            if answer and answer.strip():
                logger.info(
                    "Groq generation completed successfully."
                )

                return answer.strip()

            logger.warning(
                "Groq returned an empty response "
                "on attempt %s.",
                attempt,
            )

            if attempt <= MAX_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS)

        logger.error(
            "Groq returned empty responses after all retries."
        )

        raise RuntimeError(
            "The LLM returned an empty response after "
            "multiple attempts."
        )