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
    """Get scenario from config.ini and get required parameters needed for the server and clients"""
    
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
    

def prepare_data(df) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert pandas DataFrame features to numeric format and return features (x) and labels (y)"""

    y = df["Label"].astype('float32').values

    x = df.drop(columns=["Label"]).astype('float32').values

    return x, y


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return x and y for train, eval and test dataset"""

    # Read datasets from parquet files
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