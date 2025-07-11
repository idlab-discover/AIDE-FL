from minio import Minio

from configuration.config import Config

class MinioClient(Minio):
    def __init__(self, config: Config):
        super().__init__(
            config.minio_endpoint,
            access_key=config.minio_access_key,
            secret_key=config.minio_secret_key,
            secure=False
        )
        