import logging
import os

from configuration.config import Config

class Logger:
    _config = None
    _logger = None
    
    @classmethod
    def init(cls, config: Config):
        cls._config = config
        
    @classmethod
    def get_logger(cls, name: str):
        if cls._logger:
            return cls._logger

        log_level_str = cls._config.log_level

        log_level = getattr(logging, log_level_str, logging.INFO)
        
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.propagate = False
        cls._logger = logger
        return cls._logger