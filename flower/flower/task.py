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


def client_config_changed() -> bool:
    """Read from .ini config file whether is has changed (new scenario to be trained). Trainer-logic service sets "changed" value in .ini file."""
    
    config.read(config_filename)
    if config["clientapp"]["changed"] == "1":
        print("Client config changed")
        return True
    
    print("Client config unchanged")
    return False


def load_client_config():
    """Read client config from .ini file"""
    
    scenario = json.loads(config["clientapp"]["scenario"])

    model_data = scenario["model"]
    model_config = scenario["model_config"]
    local_epochs = scenario["hyper"]["local_epochs"]
    batch_size = scenario["hyper"]["batch_size"]

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


def prepare_data(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert pandas DataFrame features to numeric format and return features (x) and labels (y)"""

    y = df["Label"].astype('float32').values

    x = df.drop(columns=["Label"]).astype('float32').values

    return x, y


x_train = None
y_train = None
x_eval = None
y_eval = None
def load_data(config_changed) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return x and y for both train and eval dataset. Load data either from memory or from file depending on whether the config has changed."""
    global x_train, y_train, x_eval, y_eval

    # Use data from memory if scenario unchanged
    if not config_changed:
        print("Datasets loaded from memory")
        return x_train, y_train, x_eval, y_eval

    # If scenario changed, read from parquet files and load to memory
    train_file = os.path.join("share", "train.parquet")
    eval_file = os.path.join("share", "eval.parquet")
    
    try:
        train_df = pd.read_parquet(train_file)
        eval_df = pd.read_parquet(eval_file)
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        raise

    x_train, y_train = prepare_data(train_df)
    x_eval, y_eval = prepare_data(eval_df)
    
    print("Datasets loaded from files")

    return x_train, y_train, x_eval, y_eval