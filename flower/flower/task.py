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

def load_server_config():
    config.read(config_filename)
       
    scenario = json.loads(config["serverapp"]["scenario"])

    scenario_name = scenario["name"]
    model_data = scenario["model"]
    fraction_fit = scenario["strategy"]["fraction_fit"]
    fraction_evaluate = scenario["strategy"]["fraction_evaluate"]
    min_available_clients = scenario["strategy"]["min_available_clients"]
    server_rounds = scenario["hyper"]["server_rounds"]
    
    local_epochs = scenario["hyper"]["local_epochs"]
    batch_size = scenario["hyper"]["batch_size"]

    print(f"Reloaded config")

    return scenario_name, model_data, fraction_fit, fraction_evaluate, min_available_clients, server_rounds, local_epochs, batch_size


def client_config_changed() -> bool:
    """Read from .ini config file whether is has changed (new training run). If changed (1), set back to 0. Trainer-logic service sets "changed" value in .ini file."""
    
    config.read(config_filename)
    if config["clientapp"]["changed"] == "1":
        print("Client config changed")

        # Set changed to 0 after checking
        config.set("clientapp", "changed", "0")
        with open(config_filename, "w") as config_file:
            config.write(config_file)
        
        return True
    
    print("Client config unchanged")
    return False
    

def prepare_data(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert pandas DataFrame features to numeric format and return features (x) and labels (y)"""

    y = df["Label"].astype('float32').values

    x = df.drop(columns=["Label"]).astype('float32').values

    return x, y


x_train = y_train = x_eval = y_eval = x_test = y_test = None
def load_data(new_training_run) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return x and y for both train and eval dataset. Load data either from memory or from file depending on whether a new training run started meaning a new dataset needs to be loaded."""
    global x_train, y_train, x_eval, y_eval, x_test, y_test

    # Use data from memory if this is next round of an ongoing training run
    if not new_training_run:
        print("Datasets loaded from memory")
        return x_train, y_train, x_eval, y_eval, x_test, y_test

    # If new training run, read from parquet files and load to memory
    train_file = os.path.join("share", "train.parquet")
    eval_file = os.path.join("share", "eval.parquet")
    test_file = os.path.join("share", "test.parquet")
    
    try:
        train_df = pd.read_parquet(train_file)
        eval_df = pd.read_parquet(eval_file)
        test_df = pd.read_parquet(test_file)
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        raise

    x_train, y_train = prepare_data(train_df)
    x_eval, y_eval = prepare_data(eval_df)
    x_test, y_test = prepare_data(test_df)
    
    print("Datasets loaded from files")

    return x_train, y_train, x_eval, y_eval, x_test, y_test