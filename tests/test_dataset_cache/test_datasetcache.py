import pytest
from fastapi.testclient import TestClient
import pandas as pd
from pathlib import Path

def test_cache_kaggle_dataset(dataset_client, test_minio_client):
    # Test dataset caching
    response = dataset_client.post("/cache/kaggle/username/dataset")
    assert response.status_code == 200
    assert response.json()["message"] == "Successfully cached dataset username/dataset"

def test_get_dataset(dataset_client):
    # Test dataset retrieval
    response = dataset_client.get("/dataset/test_dataset")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict) 