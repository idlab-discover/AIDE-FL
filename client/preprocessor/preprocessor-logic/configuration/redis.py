from redis import Redis

from configuration.config import Config

class RedisClient:
    def __init__(self, config: Config):
        host, port = tuple(config.redis_address.split(':'))
        self.client = Redis(host=host, port=port, db=0)