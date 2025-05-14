import logging

from configuration.config import Config

class Logger:
    _config = None
    _loggers = {}
    
    @classmethod
    def init(cls, config: Config):
        cls._config = config
        
    @classmethod
    def get_logger(cls, name: str):
        if name in cls._loggers:
            return cls._loggers[name]
        
        if cls._config is None:
            raise ValueError("Logger must be initialized with a config before use")

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
        cls._loggers[name] = logger
        return logger