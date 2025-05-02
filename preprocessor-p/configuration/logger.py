import logging
import os

from config import Config

class Logger:
    _logger = None
    
    def __init__(self, config: Config):
        self.config = config
        
    @classmethod
    def get_logger(cls, name: str):
        if cls._logger:
            return cls._logger
        
        logger = logging.getLogger(name)
        
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.propagate = False
        cls._logger == logger
        return cls._logger