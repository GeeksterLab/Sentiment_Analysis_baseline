"""Tests de base de l'API.

Le modèle de machine learning est simulé dans les tests afin que la CI reste
rapide et ne dépende ni des poids du modèle ni d'un téléchargement externe.
"""

import sys
from types import ModuleType

from fastapi.testclient import TestClient


# ``api.utils`` télécharge des ressources NLTK lors de son import. L'API est
# testée ici avec un substitut léger pour ne pas rendre la CI dépendante du
# réseau ou du chargement du modèle TensorFlow.
fake_utils = ModuleType("api.utils")
fake_utils.predict_sentiment = lambda _text: {
    "prediction": 1,
    "probability": 0.5,
    "confiance": 0.5,
}
sys.modules["api.utils"] = fake_utils

from api.app import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "OK"
    assert response.json()["app"] == "Review Sentiment Analysis"


def test_predict_positive_sentiment(monkeypatch):
    def fake_predict_sentiment(text: str) -> dict:
        assert text == "This product is excellent"
        return {"prediction": 2, "probability": 0.946, "confiance": 0.946}

    monkeypatch.setattr("api.models.predict_sentiment", fake_predict_sentiment)

    response = client.post(
        "/predict",
        json={"text": "This product is excellent"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 2,
        "label": "Positive",
        "probability": 0.95,
        "confiance": 0.946,
    }


def test_predict_rejects_missing_text():
    response = client.post("/predict", json={})

    assert response.status_code == 422


def test_upload_csv_accepts_a_csv_file():
    response = client.post(
        "/upload-csv",
        files={"file": ("reviews.csv", "review\nGreat product\n", "text/csv")},
    )

    assert response.status_code == 200
    assert response.json() == []
