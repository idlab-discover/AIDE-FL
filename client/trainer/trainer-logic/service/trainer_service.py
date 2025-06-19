import configparser
import os
import json

from configuration.logger import Logger


class TrainerService:
    def __init__(self):
        self.logger = Logger.get_logger("TrainerService")
        self.shared_config = configparser.ConfigParser()
        self.config_filename = os.path.join("share", "config.ini")


    def set_flower_config(self, scenario: dict):
        self.logger.info("Setting Flower config")
        
        # Add clientapp section if not exists, otherwise ignore
        try:
            self.shared_config.add_section("clientapp")
        except (configparser.DuplicateSectionError):
            pass
        
        # Setting changed to 1 lets flower-clientapp know that this is the beginning of a new training run
        self.shared_config.set("clientapp", "changed", "1")
        
        # The default umask is 0o22 which turns off write permission of group and others
        os.umask(0)

        descriptor = os.open(
            path=self.config_filename,
            flags=(
                os.O_WRONLY  # access mode: write only
                | os.O_CREAT  # create if not exists
                | os.O_TRUNC  # truncate the file to zero
            ),
            mode=0o777
        )

        with open(descriptor, 'w') as configfile:
            self.shared_config.write(configfile)
            # the descriptor is automatically closed when configfile is closed

