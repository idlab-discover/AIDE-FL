from configuration.minio import MinioClient
from configuration.config import Config
from configuration.logger import Logger

class MinioRepository:
    def __init__(self, minio_client: MinioClient, config: Config):
        self.client = minio_client
        self.config = config
        self.logger = Logger.get_logger("MinioRepository")
    
    def insert(self, src_filename: str, dst_filename: str):
        self.logger.info(f"Uploading {src_filename} to Minio")
        self.client.fput_object(self.config.minio_bucket, dst_filename, src_filename)
        
    def get_object(self, object_name: str, dst_path: str):
        self.logger.info(f"Getting Minio object {object_name}")
        self.client.fget_object(self.config.minio_bucket, object_name, dst_path)