import pytest
from fastapi.testclient import TestClient
import sys
import os

# Ensure src/ is in the path so we can import api
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "service" in response.json()
    assert response.json()["service"] == "EdgeGuard-AI Inference API"

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "ok"
    assert "model_loaded" in data
    assert "pipeline_running" in data

def test_pipeline_status():
    response = client.get("/pipeline/status")
    assert response.status_code == 200
    data = response.json()
    assert "running" in data
    assert "status" in data
    assert "message" in data

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    # Metrics should be a plain text response with prometheus format
    assert "api_requests_total" in response.text
