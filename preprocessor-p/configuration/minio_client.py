from minio import Minio
from config import Config

class MinioClient:
    def __init__(self, config: Config):
        self.client = Minio(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=False
        )
        