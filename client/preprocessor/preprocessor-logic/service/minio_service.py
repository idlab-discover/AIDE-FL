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
        self.minio_repository.download_object(dataset_filename, dataset_filename)
        return dataset_filename
    
    
    def upload_split_dataset(self, train_path: str, eval_path: str, test_path: str, dataset_name: str, scenario_name: str):
        self.logger.info(f"Uploading split dataset {dataset_name} for {scenario_name}")
        
        train_dst = os.path.join(dataset_name, f"{scenario_name}_train.parquet")
        eval_dst = os.path.join(dataset_name, f"{scenario_name}_eval.parquet")
        test_dst = os.path.join(dataset_name, f"{scenario_name}_test.parquet")
        self.minio_repository.upload_object(train_path, train_dst)
        self.minio_repository.upload_object(eval_path, eval_dst)
        self.minio_repository.upload_object(test_path, test_dst)
        