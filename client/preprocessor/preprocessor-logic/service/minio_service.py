import os

from configuration.logger import Logger
from repository.minio_repository import MinioRepository

class MinioService:
    def __init__(self, minio_repository: MinioRepository):
        self.logger = Logger.get_logger("MinioService")
        self.minio_repository = minio_repository


    def download_dataset(self, dataset_name: str, from_server: bool = False) -> str:
        self.logger.info(f"Downloading dataset '{dataset_name}'")

        dataset_filename = dataset_name + ".parquet"
        self.minio_repository.download_dataset(dataset_filename, dataset_filename, from_server)
        return dataset_filename
        