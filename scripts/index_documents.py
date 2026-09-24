from pathlib import Path

from app.ingestion.chunker import chunk_documents
from app.ingestion.pdf_loader import load_pdf
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PDF_PATH = Path(
    "data/raw/Meditazioni_Santo_Rosario_Breve_IT_3f96a7fcaa.pdf"
)


def main() -> None:
    print("1. Loading PDF...")

    documents = load_pdf(str(PDF_PATH))

    print(f"   Pages loaded: {len(documents)}")

    print("2. Creating chunks...")

    chunks = chunk_documents(documents)

    print(f"   Chunks created: {len(chunks)}")

    print("3. Generating embeddings...")

    embedding_service = EmbeddingService(MODEL_NAME)

    embeddings = embedding_service.encode(
        [chunk["text"] for chunk in chunks]
    )

    print(f"   Embeddings generated: {len(embeddings)}")
    print(f"   Vector dimension: {len(embeddings[0])}")

    print("4. Creating vector store...")

    vector_store = VectorStore()

    vector_store.create_collection()

    print("5. Indexing documents...")

    vector_store.add_documents(
        chunks=chunks,
        embeddings=embeddings,
    )

    print("6. Indexing completed successfully.")


if __name__ == "__main__":
    main()