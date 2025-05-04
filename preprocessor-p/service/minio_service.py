import os

from repository.minio_repository import MinioRepository

class MinioService:
    def __init__(self, minio_repository: MinioRepository):
        self.minio_repository = minio_repository

    def download_dataset(self, dataset: str):
        self.minio_repository.get_object(os.path.join(dataset, dataset+".parquet"), dataset+".parquet")