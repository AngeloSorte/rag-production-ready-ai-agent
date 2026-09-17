from pathlib import Path

import pymupdf


def load_pdf(file_path: str) -> list[dict]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {path}")

    documents = []

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                documents.append(
                    {
                        "text": text,
                        "metadata": {
                            "source": path.name,
                            "page": page_number,
                        },
                    }
                )

    return documents