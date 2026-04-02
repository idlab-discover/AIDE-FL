"""flower: A Flower / TensorFlow app."""

import os
import json
import configparser
import numpy as np

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
    framework = scenario["framework"]
    if framework != "tensorflow":
        raise ValueError(f"unsupported framework: {framework}")

    scenario_name = scenario["name"]
    hold_out_dataset = scenario["hold_out_dataset"]
    drop_cols = scenario["preprocessing"]["drop_columns"]
    model_data = scenario["tensorflow"]["model"]["config"]
    fraction_fit = scenario["aggregation"]["fraction_fit"]
    fraction_evaluate = scenario["aggregation"]["fraction_evaluate"]
    min_available_clients = scenario["aggregation"]["min_available_clients"]
    server_rounds = scenario["training"]["server_rounds"]
    
    local_epochs = scenario["training"]["local_epochs"]
    batch_size = scenario["training"]["batch_size"]
    label_column = scenario["label_column"]

    print(f"Reloaded config")

    return framework, scenario_name, hold_out_dataset, drop_cols, model_data, fraction_fit, fraction_evaluate, min_available_clients, server_rounds, local_epochs, batch_size, label_column
    

def prepare_data(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert pandas DataFrame features to numeric format and return features (x) and labels (y). Name of label column needs to be provided."""

    y = df[label_column].astype('float32').values

    x = df.drop(columns=[label_column]).astype('float32').values

    return x, y


def get_model_input_shape(model_data: dict) -> list:
    """Return the model input layer batch shape."""

    layers = model_data["config"]["layers"]
    if not layers:
        raise ValueError("model does not define any layers")

    input_config = layers[0]["config"]
    input_shape = input_config.get("batch_shape")
    if input_shape is None:
        input_shape = input_config.get("batch_input_shape")
    if input_shape is None:
        raise ValueError("model input layer is missing batch_shape")

    return input_shape


def adapt_features_to_model_input(x: np.ndarray, model_data: dict) -> np.ndarray:
    """Adapt tabular features to the model's expected input rank."""

    input_shape = get_model_input_shape(model_data)
    if len(input_shape) == 2:
        return x

    if len(input_shape) == 3 and input_shape[2] == 1:
        return np.expand_dims(x, axis=-1)

    raise ValueError(f"Unsupported model input shape: {input_shape}")


def load_server_data(label_column: str, drop_cols) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return x and y for hold-out dataset named 'hold-out.parquet' in shared folder"""

    # Read dataset from parquet file
    hold_out_file = os.path.join("share", "hold-out.parquet")
    
    try:
        hold_out_df = pd.read_parquet(hold_out_file)
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        raise

    hold_out_df = drop_columns(hold_out_df, drop_cols)

    x_hold_out, y_hold_out = prepare_data(hold_out_df, label_column)
    
    print("Hold-out dataset loaded from file")

    return x_hold_out, y_hold_out


def load_data(label_column: str, model_data: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return x and y for train, eval and test dataset named 'train.parquet', 'eval.parquet', 'test.parquet' in shared folder"""

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

    x_train, y_train = prepare_data(train_df, label_column)
    x_eval, y_eval = prepare_data(eval_df, label_column)
    x_test, y_test = prepare_data(test_df, label_column)
    x_train = adapt_features_to_model_input(x_train, model_data)
    x_eval = adapt_features_to_model_input(x_eval, model_data)
    x_test = adapt_features_to_model_input(x_test, model_data)
    
    print("Datasets loaded from files")

    return x_train, y_train, x_eval, y_eval, x_test, y_test


def drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    print(f"Dropping columns: {cols}")
    return df.drop(columns=cols)
