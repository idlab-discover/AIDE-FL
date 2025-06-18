from configuration.redis import RedisClient
from configuration.config import Config
from configuration.logger import Logger

class RedisRepository:
    def __init__(self, redis_client: RedisClient, config: Config):
        self.redis = redis_client.client
        self.config = config
        self.logger = Logger.get_logger("RedisRepository")
        
        
    def download(self, key: str):
        self.logger.info(f"Downloading {key}")
        return self.redis.get(key)