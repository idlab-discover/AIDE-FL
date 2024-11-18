import pytest
import os
from fastapi.testclient import TestClient
import tempfile
import shutil

@pytest.fixture
def test_model_dir():
    # Create a temporary directory for model storage
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after tests
    shutil.rmtree(temp_dir)

def test_upload_model(model_client, test_model_dir):
    # Test model upload
    test_model_content = b"dummy model content"
    files = {
        "model_file": ("test_model.onnx", test_model_content)
    }
    response = model_client.post(
        "/models/upload/test_experiment",
        files=files,
        data={"model_type": "onnx", "description": "Test model"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["experiment_id"] == "test_experiment"
    assert data["model_type"] == "onnx"
    assert data["version"] == 1

def test_get_model_versions(model_client):
    response = model_client.get("/models/test_experiment/versions")
    assert response.status_code in [200, 404]  # 404 if no models exist yet

def test_get_latest_model(model_client):
    response = model_client.get("/models/test_experiment/latest")
    assert response.status_code in [200, 404]  # 404 if no models exist yet

def test_delete_model(model_client):
    # First upload a model
    test_model_content = b"dummy model content"
    files = {
        "model_file": ("test_model.onnx", test_model_content)
    }
    upload_response = model_client.post(
        "/models/upload/test_experiment",
        files=files,
        data={"model_type": "onnx"}
    )
    assert upload_response.status_code == 200
    
    # Then delete it
    version = upload_response.json()["version"]
    delete_response = model_client.delete(f"/models/test_experiment/{version}")
    assert delete_response.status_code == 200 