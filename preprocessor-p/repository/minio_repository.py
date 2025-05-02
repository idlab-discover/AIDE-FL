import logging


from configuration.minio_client import MinioClient
from configuration.config import Config

class MinioRepository:
    def __init__(self, minio_client: MinioClient, config: Config):
        self.client = minio_client
        self.config = config
    
    def insert(self, src_filename: str, dst_filename: str):
        self.client.fput_object(self.config.minio_bucket, dst_filename, src_filename)
        
    def get_object(self, object_name: str, dst_path: str):
        self.client.fget_object(self.config.minio_bucket, object_name, dst_path)