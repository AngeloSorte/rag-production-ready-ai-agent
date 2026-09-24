from app.generation.prompt_builder import build_prompt
from app.retrieval.retriever import Retriever


def main() -> None:
    query = "Che cosa meditiamo nel Santo Rosario?"

    retriever = Retriever()

    documents = retriever.retrieve(
        query,
        retrieval_limit=10,
        top_k=5,
    )

    prompt = build_prompt(
        query=query,
        documents=documents,
    )

    print(prompt)


if __name__ == "__main__":
    main()