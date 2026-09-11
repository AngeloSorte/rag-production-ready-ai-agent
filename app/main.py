from fastapi import FastAPI

app = FastAPI(
    title="RAG Production-Ready AI Agent",
    version="0.1.0",
    description="Production-ready RAG AI Agent API.",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}