from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"


def test_ask_with_valid_question() -> None:
    response = client.post(
        "/api/ask",
        json={
            "question": "Che cosa meditiamo nel Santo Rosario?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "citations" in data

    assert isinstance(data["answer"], str)
    assert data["answer"].strip()

    assert isinstance(data["citations"], list)


def test_ask_with_empty_question() -> None:
    response = client.post(
        "/api/ask",
        json={
            "question": ""
        },
    )

    assert response.status_code == 422