"""
Ce fichier contient des tests unitaires pour vérifier le bon fonctionnement des endpoints de prédiction et de reload de l'API
"""


from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API Rakuten active"}

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}