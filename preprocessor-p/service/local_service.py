import os
from pathlib import Path

from configuration.logger import Logger

class LocalService:
    def __init__(self, base_path: str):
        self.logger = Logger.get_logger("LocalService")
        self.base_path = Path(base_path)

    def exists(self, file: str) -> bool:
        return (self.base_path / file).exists()
    
    def delete(self, file: str):
        self.logger.info(f"Deleting {file}")

        path = self.base_path / file
        if path.exists():
            path.unlink()
