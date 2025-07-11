from configuration.minio import MinioClient, MinioServerClient
from configuration.config import Config
from configuration.logger import Logger

class MinioRepository:
    def __init__(self, minio_client: MinioClient, minio_server_client: MinioServerClient, config: Config):
        self.minio = minio_client
        self.minio_server = minio_server_client
        self.config = config
        self.logger = Logger.get_logger("MinioRepository")
    
    def upload_dataset(self, src_filename: str, dst_filename: str, from_server: bool = False):
        if from_server:
            self.logger.info(f"Uploading dataset '{src_filename}' to server Minio")
            self.minio_server.fput_object(self.config.minio_server_datasets_bucket, dst_filename, src_filename)
            return
        
        self.logger.info(f"Uploading dataset '{src_filename}' to Minio")
        self.minio.fput_object(self.config.minio_datasets_bucket, dst_filename, src_filename)
        
    def download_dataset(self, object_name: str, dst_path: str, from_server: bool = False):
        if from_server:
            self.logger.info(f"Getting Minio dataset '{object_name}' from server")
            self.minio_server.fget_object(self.config.minio_server_datasets_bucket, object_name, dst_path)
            return
        
        self.logger.info(f"Getting Minio dataset '{object_name}'")
        self.minio.fget_object(self.config.minio_datasets_bucket, object_name, dst_path)
        
    