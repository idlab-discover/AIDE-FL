"""Shared scenario and parquet helpers for the AIDE-FL Flower app."""

import os
import json
import configparser
import numpy as np
from collections.abc import Callable

import pandas as pd


config_filename = os.path.join("share", "config.ini")
config = configparser.ConfigParser()

def load_server_config() -> dict:
    """Load the rendered scenario from the shared Flower config file."""

    config.read(config_filename)
    scenario = json.loads(config["serverapp"]["scenario"])
    print("Reloaded config")
    return scenario


def prepare_data(df: pd.DataFrame, label_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert pandas DataFrame features to numeric format and return features and labels."""

    y = df[label_column].astype("float32").values
    x = df.drop(columns=[label_column]).astype("float32").values
    return x, y


def load_server_data(
    label_column: str,
    drop_cols,
    adapt_features: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return hold-out features and labels from the shared parquet dataset."""

    hold_out_file = os.path.join("share", "hold-out.parquet")
    try:
        hold_out_df = pd.read_parquet(hold_out_file)
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        raise

    hold_out_df = drop_columns(hold_out_df, drop_cols)
    x_hold_out, y_hold_out = prepare_data(hold_out_df, label_column)
    if adapt_features is not None:
        x_hold_out = adapt_features(x_hold_out)

    print("Hold-out dataset loaded from file")
    return x_hold_out, y_hold_out


def load_data(
    label_column: str,
    adapt_features: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/eval/test features and labels from the shared parquet datasets."""

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
    if adapt_features is not None:
        x_train = adapt_features(x_train)
        x_eval = adapt_features(x_eval)
        x_test = adapt_features(x_test)

    print("Datasets loaded from files")
    return x_train, y_train, x_eval, y_eval, x_test, y_test


def drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    print(f"Dropping columns: {cols}")
    return df.drop(columns=cols)
