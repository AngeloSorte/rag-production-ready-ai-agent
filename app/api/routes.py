import logging
from typing import Any

from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import Field

from app.generation.generator import Generator
from app.generation.llm import GroqLLM
from app.retrieval.retriever import Retriever


logger = logging.getLogger(__name__)

router = APIRouter()

retriever = Retriever()

generator = Generator(
    llm=GroqLLM(),
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        description="Question to answer using the indexed documents.",
    )


class AskResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]


@router.post(
    "/ask",
    response_model=AskResponse,
)
def ask(request: AskRequest) -> AskResponse:
    logger.info(
        "Received /ask request question_length=%s",
        len(request.question),
    )

    try:
        documents = retriever.retrieve(
            request.question,
            retrieval_limit=10,
            top_k=5,
        )

        logger.info(
            "Retrieved %s documents for /ask request",
            len(documents),
        )

        result = generator.generate(
            query=request.question,
            documents=documents,
        )

        logger.info(
            "Successfully generated /ask response",
        )

        return AskResponse(**result)

    except RuntimeError as exc:
        logger.error(
            "Generation service error: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc