from typing import Any

from app.generation.generator import Generator
from app.generation.llm import GroqLLM
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.retriever import Retriever


EMBEDDING_MODEL = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

SIMILARITY_THRESHOLD = 0.60


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    dot_product = sum(
        a * b
        for a, b in zip(vector_a, vector_b)
    )

    norm_a = sum(a * a for a in vector_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def split_into_sentences(text: str) -> list[str]:
    sentences = []

    for sentence in text.replace("!", ".").replace("?", ".").split("."):
        sentence = sentence.strip()

        if sentence:
            sentences.append(sentence)

    return sentences


def calculate_grounding_score(
    answer: str,
    documents: list[dict[str, Any]],
    embedding_service: EmbeddingService,
) -> float:
    sentences = split_into_sentences(answer)

    if not sentences:
        return 0.0

    context = [
        document["text"]
        for document in documents
        if document.get("text")
    ]

    if not context:
        return 0.0

    sentence_embeddings = embedding_service.encode(sentences)
    context_embeddings = embedding_service.encode(context)

    supported = 0

    for sentence_embedding in sentence_embeddings:
        similarities = [
            cosine_similarity(
                sentence_embedding,
                context_embedding,
            )
            for context_embedding in context_embeddings
        ]

        best_similarity = max(similarities)

        if best_similarity >= SIMILARITY_THRESHOLD:
            supported += 1

    return supported / len(sentences)


def main() -> None:
    question = "Che cosa meditiamo nel Santo Rosario?"

    retriever = Retriever()
    generator = Generator(
        llm=GroqLLM(),
    )
    embedding_service = EmbeddingService(EMBEDDING_MODEL)

    print("=== GROUNDING EVALUATION ===\n")

    documents = retriever.retrieve(
        question,
        retrieval_limit=10,
        top_k=5,
    )

    result = generator.generate(
        query=question,
        documents=documents,
    )

    answer = result["answer"]

    score = calculate_grounding_score(
        answer=answer,
        documents=documents,
        embedding_service=embedding_service,
    )

    print("Question:")
    print(question)
    print()

    print("Answer:")
    print(answer)
    print()

    print(f"Grounding score: {score:.2%}")


if __name__ == "__main__":
    main()