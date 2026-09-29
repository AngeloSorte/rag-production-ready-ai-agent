import json
from pathlib import Path

from app.retrieval.retriever import Retriever


TEST_CASES_PATH = Path("tests/evaluation/test_cases.json")


def load_test_cases() -> list[dict]:
    with TEST_CASES_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def check_hit_at_k(
    results: list[dict],
    expected_sources: list[dict],
    k: int,
) -> bool:
    top_results = results[:k]

    for result in top_results:
        metadata = result["metadata"]

        for expected in expected_sources:
            source_matches = (
                metadata["source"] == expected["source"]
            )

            page_matches = (
                "page" not in expected
                or metadata["page"] == expected["page"]
            )

            if source_matches and page_matches:
                return True

    return False


def main() -> None:
    test_cases = load_test_cases()
    retriever = Retriever()

    answerable_cases = [
        test_case
        for test_case in test_cases
        if test_case.get("answerable", True)
    ]

    unanswerable_cases = [
        test_case
        for test_case in test_cases
        if not test_case.get("answerable", True)
    ]

    total_answerable = len(answerable_cases)
    passed = 0

    print("=== RETRIEVAL EVALUATION ===\n")

    for index, test_case in enumerate(answerable_cases, start=1):
        question = test_case["question"]
        expected_sources = test_case["expected_sources"]

        results = retriever.retrieve(
            question,
            retrieval_limit=10,
            top_k=5,
        )

        hit = check_hit_at_k(
            results=results,
            expected_sources=expected_sources,
            k=5,
        )

        if hit:
            passed += 1

        status = "PASS" if hit else "FAIL"

        print(f"[{index}/{total_answerable}] {status}")
        print(f"Question: {question}")

        print("Retrieved:")
        for result in results:
            metadata = result["metadata"]
            print(
                f"  - {metadata['source']} "
                f"(page {metadata['page']}, "
                f"chunk {metadata['chunk_index']})"
            )

        print()

    hit_rate = passed / total_answerable if total_answerable else 0.0

    print("=== SUMMARY ===")
    print(f"Answerable cases: {passed}/{total_answerable}")
    print(f"Hit@5: {hit_rate:.2%}")

    if unanswerable_cases:
        print()
        print("=== UNANSWERABLE CASES ===")

        for test_case in unanswerable_cases:
            print(f"- {test_case['question']}")

        print(
            "\nThese cases are evaluated by the generation/"
            "grounding tests, not by Hit@5."
        )


if __name__ == "__main__":
    main()