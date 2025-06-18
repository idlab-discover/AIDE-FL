from configuration.redis import RedisClient
from configuration.config import Config
from configuration.logger import Logger

class RedisRepository:
    def __init__(self, redis_client: RedisClient, config: Config):
        self.redis = redis_client.client
        self.config = config
        self.logger = Logger.get_logger("RedisRepository")
        
        
    def upload(self, key: str, value):
        self.logger.info(f"Setting {key}")
        self.redis.set(key, value, ex=int(self.config.redis_exp))
        
        
    def exists(self, *key: str) -> int:
        self.logger.info(f"Checking if {key} exists")
        return self.redis.exists(*key)
    
    def reset_ttl(self, key: str):
        self.logger.info(f"Resetting TTL of {key} to {self.config.redis_exp}")
        self.redis.expire(key, int(self.config.redis_exp))