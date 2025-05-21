"""flwr-keras: A Flower / TensorFlow app."""

import os
import json
import configparser

from tensorflow import keras


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

config_filename = os.path.join("share", "config.ini")
config = configparser.ConfigParser()

model_config_data = None
model = None
fraction_fit = None
fraction_evaluate = None
min_available_clients = None
server_rounds = None


def reload_config():
    global model, model_config_data, fraction_evaluate, fraction_fit, min_available_clients, server_rounds
    # Check whether scenario has changed and extract its data
    config.read(config_filename)
    if config["serverapp"]["changed"] == "1":
        scenario = json.load(config["serverapp"]["scenario"])

        model_data = scenario["model"]
        model_config_data = scenario["model_config"]
        fraction_fit = scenario["strategy"]["fraction_fit"]
        fraction_evaluate = scenario["strategy"]["fraction_evaluate"]
        min_available_clients = scenario["strategy"]["min_available_clients"]
        server_rounds = scenario["hyper"]["server_rounds"]

        # Load model from json
        model = keras.models.model_from_json(json.dumps(model_data))
        print(f"Reloaded model architecture")

        # Set changed to 0
        config.set("serverapp", "changed", "0")
        with open(config_filename, "w") as config_file:
            config.write(config_file)


def load_model():
    # Retrieve model config
    optimizer = model_config_data["optimizer"]
    loss = model_config_data["loss"]
    metrics = model_config_data["metrics"]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("Model compiled")

    return model
