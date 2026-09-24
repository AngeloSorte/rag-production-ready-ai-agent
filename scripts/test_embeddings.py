from app.retrieval.embeddings import EmbeddingService


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def main() -> None:
    service = EmbeddingService(MODEL_NAME)

    texts = [
        "La preghiera del Santo Rosario è una meditazione cristiana.",
        "Il Rosario è una forma di preghiera cristiana.",
        "Python è un linguaggio di programmazione.",
    ]

    embeddings = service.encode(texts)

    print(f"Texts: {len(texts)}")
    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"First vector values: {embeddings[0][:5]}")


if __name__ == "__main__":
    main()