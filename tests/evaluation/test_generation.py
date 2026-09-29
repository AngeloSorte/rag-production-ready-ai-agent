import json
import re
from pathlib import Path

from app.generation.generator import Generator
from app.generation.llm import GroqLLM
from app.retrieval.retriever import Retriever


TEST_CASES_PATH = Path(
    "tests/evaluation/test_generation_cases.json"
)


def load_test_cases() -> list[dict]:
    with TEST_CASES_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def normalize_text(text: str) -> str:
    text = text.lower()

    text = (
        text.replace("à", "a")
        .replace("è", "e")
        .replace("é", "e")
        .replace("ì", "i")
        .replace("ò", "o")
        .replace("ù", "u")
    )

    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def fact_is_covered(
    fact: str,
    answer: str,
) -> bool:
    normalized_fact = normalize_text(fact)
    normalized_answer = normalize_text(answer)

    fact_words = set(normalized_fact.split())
    answer_words = set(normalized_answer.split())

    if not fact_words:
        return False

    overlap = fact_words.intersection(answer_words)

    coverage = len(overlap) / len(fact_words)

    return coverage >= 0.60


def forbidden_term_is_present(
    term: str,
    answer: str,
) -> bool:
    normalized_term = normalize_text(term)
    normalized_answer = normalize_text(answer)

    return normalized_term in normalized_answer


def is_refusal(answer: str) -> bool:
    refusal_markers = [
        "non contiene informazioni sufficienti",
        "non contiene abbastanza informazioni",
        "non ho informazioni sufficienti",
        "non posso rispondere",
        "non è possibile rispondere",
        "documents provided do not contain",
        "provided documents do not contain",
        "not enough information",
        "insufficient information",
    ]

    normalized_answer = normalize_text(answer)

    return any(
        normalize_text(marker) in normalized_answer
        for marker in refusal_markers
    )


def evaluate_answer(
    answer: str,
    answerable: bool,
    expected_facts: list[str],
    forbidden_terms: list[str],
) -> tuple[bool, list[str]]:
    failures = []

    if not answer.strip():
        return False, ["Empty answer"]

    if not answerable:
        if not is_refusal(answer):
            failures.append(
                "Expected a refusal because the question "
                "is not answerable from the documents."
            )

        return not failures, failures

    for fact in expected_facts:
        if not fact_is_covered(
            fact,
            answer,
        ):
            failures.append(
                f"Missing expected fact: {fact}"
            )

    for term in forbidden_terms:
        if forbidden_term_is_present(
            term,
            answer,
        ):
            failures.append(
                f"Forbidden content detected: {term}"
            )

    return not failures, failures


def main() -> None:
    test_cases = load_test_cases()

    retriever = Retriever()
    generator = Generator(
        llm=GroqLLM(),
    )

    total = len(test_cases)
    passed = 0

    print("=== GENERATION EVALUATION ===\n")

    for index, test_case in enumerate(
        test_cases,
        start=1,
    ):
        question = test_case["question"]
        answerable = test_case["answerable"]
        expected_facts = test_case["expected_facts"]
        forbidden_terms = test_case.get(
            "forbidden_terms",
            [],
        )

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

        passed_case, failures = evaluate_answer(
            answer=answer,
            answerable=answerable,
            expected_facts=expected_facts,
            forbidden_terms=forbidden_terms,
        )

        if passed_case:
            passed += 1

        status = "PASS" if passed_case else "FAIL"

        print(f"[{index}/{total}] {status}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")

        if failures:
            print("Evaluation issues:")

            for failure in failures:
                print(f"  - {failure}")

        print()

    score = passed / total if total else 0.0

    print("=== SUMMARY ===")
    print(f"Passed: {passed}/{total}")
    print(f"Generation score: {score:.2%}")


if __name__ == "__main__":
    main()