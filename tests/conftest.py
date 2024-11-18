import pytest
from minio import Minio
from fastapi.testclient import TestClient
from aide_fl.dataset_cache.datasetcache import app as dataset_app
from aide_fl.model_server.modelserver import app as model_app

@pytest.fixture
def test_minio_client():
    return Minio(
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False
    )

@pytest.fixture
def dataset_client():
    return TestClient(dataset_app)

@pytest.fixture
def model_client():
    return TestClient(model_app) 