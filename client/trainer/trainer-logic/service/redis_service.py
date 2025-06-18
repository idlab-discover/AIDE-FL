from configuration.logger import Logger
from repository.redis_repository import RedisRepository

import pandas as pd
import pyarrow.parquet as pq
import os

class RedisService:
    def __init__(self, redis_repository: RedisRepository):
        self.redis_repo = redis_repository
        self.logger = Logger.get_logger("RedisService")
        
    
    def download_datasets_to_parquet(self, scenario_name: str):
        self.logger.info(f"Downloading datasets for scenario '{scenario_name}'")
        
        parquet_bytes = self.redis_repo.download(f"{scenario_name}:train")
        with open(os.path.join("share", "train.parquet"), 'wb') as f:
            f.write(parquet_bytes)
        
        parquet_bytes = self.redis_repo.download(f"{scenario_name}:eval")
        with open(os.path.join("share", "eval.parquet"), 'wb') as f:
            f.write(parquet_bytes)
            
        parquet_bytes = self.redis_repo.download(f"{scenario_name}:test")
        with open(os.path.join("share", "test.parquet"), 'wb') as f:
            f.write(parquet_bytes)