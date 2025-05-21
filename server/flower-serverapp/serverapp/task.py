"""flower-serverapp: A Flower / TensorFlow app."""

import os
import json
import configparser

from tensorflow import keras


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

config_filename = os.path.join("share", "config.ini")
config = configparser.ConfigParser()

gl_model_config_data = None
gl_model = None
gl_fraction_fit = None
gl_fraction_evaluate = None
gl_min_available_clients = None
gl_server_rounds = None


def reload_config():
    global gl_model, gl_model_config_data, gl_fraction_evaluate, gl_fraction_fit, gl_min_available_clients, gl_server_rounds
    # Check whether scenario has changed and extract its data
    config.read(config_filename)
    if config["serverapp"]["changed"] == "1":
        scenario = json.load(config["serverapp"]["scenario"])

        model_data = scenario["model"]
        gl_model_config_data = scenario["model_config"]
        gl_fraction_fit = scenario["strategy"]["fraction_fit"]
        gl_fraction_evaluate = scenario["strategy"]["fraction_evaluate"]
        gl_min_available_clients = scenario["strategy"]["min_available_clients"]
        gl_server_rounds = scenario["hyper"]["server_rounds"]

        # Load model from json
        gl_model = keras.models.model_from_json(json.dumps(model_data))
        print(f"Reloaded model architecture")

        # Set changed to 0
        config.set("serverapp", "changed", "0")
        with open(config_filename, "w") as config_file:
            config.write(config_file)


def load_model():
    # Retrieve model config
    optimizer = gl_model_config_data["optimizer"]
    loss = gl_model_config_data["loss"]
    metrics = gl_model_config_data["metrics"]
    
    gl_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("Model compiled")

    return gl_model
