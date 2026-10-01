FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10
ENV HF_HOME=/opt/huggingface

COPY pyproject.toml .
COPY README.md .
COPY app ./app
COPY data/qdrant ./data/qdrant

RUN pip install --no-cache-dir --upgrade pip

RUN pip install \
    --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install \
    --no-cache-dir \
    --retries 10 \
    --timeout 120 \
    .

RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); CrossEncoder('vincolle/reranker-bert-italian-uncased-mmarco-mnrl')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]