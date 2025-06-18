from repository.redis_repository import RedisRepository
from configuration.logger import Logger

import pandas as pd
import io

class RedisService:
    def __init__(self, redis_repository: RedisRepository):
        self.redis_repo = redis_repository
        self.logger = Logger.get_logger("RedisService")
        
        
    def store_dataframes_as_parquet(self, scenario_name: str, df_train: pd.DataFrame, df_eval: pd.DataFrame, df_test: pd.DataFrame):
        self.logger.info(f"Storing dataframes in Redis")
        
        train_parquet_bytes = io.BytesIO()
        df_train.to_parquet(train_parquet_bytes)
        self.redis_repo.upload(f"{scenario_name}:train", train_parquet_bytes.getvalue())

        eval_parquet_bytes = io.BytesIO()        
        df_eval.to_parquet(eval_parquet_bytes)
        self.redis_repo.upload(f"{scenario_name}:eval", eval_parquet_bytes.getvalue())
        
        test_parquet_bytes = io.BytesIO()
        df_test.to_parquet(test_parquet_bytes)
        self.redis_repo.upload(f"{scenario_name}:test", test_parquet_bytes.getvalue())
        
        
    def are_datasets_cached(self, scenario_name: str) -> bool:
        self.logger.info(f"Checking whether preprocessed datasets of scenario '{scenario_name}' are cached")
        
        cached = self.redis_repo.exists(f"{scenario_name}:train", f"{scenario_name}:eval", f"{scenario_name}:test")
        
        if cached == 3:
            self.logger.info(f"Train, eval and test datasets are cached")
            return True
        
        self.logger.info(f"Not all datasets are cached: {cached}")
        return False
    
    
    def reset_datasets_ttl(self, scenario_name: str):
        self.logger.info(f"Resetting TTL for datasets of scenario '{scenario_name}'")
        
        self.redis_repo.reset_ttl(f"{scenario_name}:train")
        self.redis_repo.reset_ttl(f"{scenario_name}:eval")
        self.redis_repo.reset_ttl(f"{scenario_name}:test")
    