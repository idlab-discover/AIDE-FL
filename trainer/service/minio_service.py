import os

from configuration.logger import Logger
from repository.minio_repository import MinioRepository

class MinioService:
    def __init__(self, minio_repository: MinioRepository):
        self.logger = Logger.get_logger("MinioService")
        self.minio_repository = minio_repository


    def download_dataset(self, dataset_name: str) -> str:
        self.logger.info(f"Downloading dataset {dataset_name}")

        dataset_filename = dataset_name + ".parquet"
        self.minio_repository.download_object(os.path.join(dataset_name, dataset_filename), dataset_filename)
        return dataset_filename
        