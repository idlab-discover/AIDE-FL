"""flower-clientapp: A Flower / TensorFlow app."""

import os
import json
import configparser

from tensorflow import keras
import pandas as pd


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

config_filename = os.path.join("share", "config.ini")
config = configparser.ConfigParser()

gl_model_config_data = None
gl_model = None
gl_local_epochs = None
gl_batch_size = None
gl_scenario_change = True


def reload_config():
    global gl_model, gl_model_config_data, gl_local_epochs, gl_batch_size, gl_scenario_change
    # Check whether scenario has changed and extract its data
    config.read(config_filename)
    if config["clientapp"]["changed"] == "1":
        scenario = json.load(config["clientapp"]["scenario"])

        model_data = scenario["model"]
        gl_model_config_data = scenario["model_config"]
        gl_local_epochs = scenario["hyper"]["local_epochs"]
        gl_batch_size = scenario["hyper"]["batch_size"]

        gl_scenario_change = True

        # Load model from json
        gl_model = keras.models.model_from_json(json.dumps(model_data))
        print(f"Reloaded model architecture")

        # Set changed to 0
        config.set("clientapp", "changed", "0")
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


def prepare_data(df):
    """Convert pandas DataFrame features to numeric format and handle label"""

    y = df["Label"].astype('float32').values

    x = df.drop(columns=["Label"]).astype('float32').values

    return x, y


x_train, y_train, x_eval, y_eval = None
def load_data():
    global x_train, y_train, x_eval, y_eval, gl_scenario_change

    train_file = os.path.join("share", "train.parquet")
    eval_file = os.path.join("share", "eval.parquet")

    # Use data from memory if scenario unchanged
    if not gl_scenario_change:
        return x_train, y_train, x_eval, y_eval

    try:
        train_df = pd.read_parquet(train_file)
        eval_df = pd.read_parquet(eval_file)
    except Exception as e:
        print(f"Error loading parquet files: {e}")
        raise

    x_train, y_train = prepare_data(train_df)
    x_eval, y_eval = prepare_data(eval_df)

    gl_scenario_change = False

    return x_train, y_train, x_eval, y_eval