"""flower: A Flower / TensorFlow app."""

import os
import json
import configparser

from tensorflow import keras
import pandas as pd


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

config_filename = os.path.join("share", "config.ini")
config = configparser.ConfigParser()

def reload_server_config():
    # Check whether scenario has changed and extract its data
    config.read(config_filename)
       
    scenario = json.loads(config["serverapp"]["scenario"])

    model_data = scenario["model"]
    model_config = scenario["model_config"]
    fraction_fit = scenario["strategy"]["fraction_fit"]
    fraction_evaluate = scenario["strategy"]["fraction_evaluate"]
    min_available_clients = scenario["strategy"]["min_available_clients"]
    server_rounds = scenario["hyper"]["server_rounds"]

    # Load model from json
    model = keras.models.model_from_json(json.dumps(model_data))
    print(f"Reloaded configliuvdfvhrsikvsdfbkluveszhdvuilsdhvklesdjhvhediulvdhilu")

    return model, model_config, fraction_fit, fraction_evaluate, min_available_clients, server_rounds


scenario_change = True
def reload_client_config():
    global scenario_change
    # Check whether scenario has changed and extract its data
    config.read(config_filename)
    if config["clientapp"]["changed"] == "1":
        scenario = json.loads(config["clientapp"]["scenario"])

        model_data = scenario["model"]
        model_config = scenario["model_config"]
        local_epochs = scenario["hyper"]["local_epochs"]
        batch_size = scenario["hyper"]["batch_size"]

        scenario_change = True

        # Load model from json
        model = keras.models.model_from_json(json.dumps(model_data))
        print(f"Reloaded model architecture")

        # Set changed to 0
        config.set("clientapp", "changed", "0")
        with open(config_filename, "w") as config_file:
            config.write(config_file)

        return model, model_config, local_epochs, batch_size


def load_model(model, model_config):
    # Retrieve model config
    optimizer = model_config["optimizer"]
    loss = model_config["loss"]
    metrics = model_config["metrics"]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    print("Model compiled")

    return model


def prepare_data(df):
    """Convert pandas DataFrame features to numeric format and handle label"""

    y = df["Label"].astype('float32').values

    x = df.drop(columns=["Label"]).astype('float32').values

    return x, y


x_train = None
y_train = None
x_eval = None
y_eval = None
def load_data():
    global x_train, y_train, x_eval, y_eval, scenario_change

    train_file = os.path.join("share", "train.parquet")
    eval_file = os.path.join("share", "eval.parquet")

    # Use data from memory if scenario unchanged
    if not scenario_change:
        return x_train, y_train, x_eval, y_eval

    try:
        train_df = pd.read_parquet(train_file)
        eval_df = pd.read_parquet(eval_file)
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        raise

    x_train, y_train = prepare_data(train_df)
    x_eval, y_eval = prepare_data(eval_df)

    scenario_change = False

    return x_train, y_train, x_eval, y_eval