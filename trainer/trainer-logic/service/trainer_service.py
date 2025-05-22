import configparser
import os
import json

class TrainerService:
    def __init__(self):
        self.shared_config = configparser.ConfigParser()
        self.config_filename = os.path.join("share", "config.ini")


    def set_flower_config(self, scenario: dict):
        self.shared_config.add_section("clientapp")
        self.shared_config.set("clientapp", "changed", "1")
        self.shared_config.set("clientapp", "scenario", json.dumps(scenario))

        with open(self.config_filename, 'w') as configfile:
            self.shared_config.write(configfile)
