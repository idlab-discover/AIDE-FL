import pytest
from minio import Minio

def test_minio_connection(test_minio_client):
    # Test MinIO connection and basic operations
    bucket_name = "test-bucket"
    
    # Create test bucket
    if not test_minio_client.bucket_exists(bucket_name):
        test_minio_client.make_bucket(bucket_name)
    
    assert test_minio_client.bucket_exists(bucket_name)

def test_minio_operations(test_minio_client):
    bucket_name = "test-bucket"
    test_object = "test.txt"
    test_data = b"Hello, MinIO!"
    
    # Test object operations
    try:
        # Upload test object
        test_minio_client.put_object(
            bucket_name,
            test_object,
            data=test_data,
            length=len(test_data)
        )
        
        # Verify object exists
        assert any(obj.object_name == test_object 
                  for obj in test_minio_client.list_objects(bucket_name))
    finally:
        # Cleanup
        test_minio_client.remove_object(bucket_name, test_object) 