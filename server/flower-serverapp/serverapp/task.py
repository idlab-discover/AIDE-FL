"""flower-serverapp: A Flower / TensorFlow app."""

import os
import json
import configparser

from tensorflow import keras


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

config_filename = os.path.join("share", "config.ini")
config = configparser.ConfigParser()

def reload_config():
    # Check whether scenario has changed and extract its data
    config.read(config_filename)
    if config["serverapp"]["changed"] == "1":
        scenario = json.load(config["serverapp"]["scenario"])

        model_data = scenario["model"]
        model_config = scenario["model_config"]
        fraction_fit = scenario["strategy"]["fraction_fit"]
        fraction_evaluate = scenario["strategy"]["fraction_evaluate"]
        min_available_clients = scenario["strategy"]["min_available_clients"]
        server_rounds = scenario["hyper"]["server_rounds"]

        # Load model from json
        model = keras.models.model_from_json(json.dumps(model_data))
        print(f"Reloaded model architecture")
        
        return model, model_config, fraction_fit, fraction_evaluate, min_available_clients, server_rounds


def load_model(model, model_config):
    # Retrieve model config
    optimizer = model_config["optimizer"]
    loss = model_config["loss"]
    metrics = model_config["metrics"]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("Model compiled")

    return model
