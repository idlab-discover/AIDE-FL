from configuration.logger import Logger

import pyarrow.parquet as pq
import json
import os

class TrainerService:
    def __init__(self):
        self.logger = Logger.get_logger("TrainerService")

    def _get_framework(self, scenario: dict) -> str:
        framework = scenario.get("framework")
        if framework:
            return framework

        if isinstance(scenario.get("tensorflow"), dict):
            return "tensorflow"
        if isinstance(scenario.get("scikit_learn"), dict):
            return "scikit_learn"
        if isinstance(scenario.get("xgboost"), dict):
            return "xgboost"

        raise ValueError("unsupported framework: None")

    def _get_tensorflow_model_config(self, scenario: dict):
        tensorflow = scenario.get("tensorflow")
        if not isinstance(tensorflow, dict):
            raise ValueError("scenario is missing tensorflow block")

        model = tensorflow.get("model")
        if not isinstance(model, dict):
            raise ValueError("scenario tensorflow block is missing model")

        model_config = model.get("config")
        if not isinstance(model_config, dict):
            raise ValueError("scenario tensorflow model is missing config")

        return model_config

    def _get_model_config(self, scenario: dict):
        framework = self._get_framework(scenario)
        if framework == "tensorflow":
            return self._get_tensorflow_model_config(scenario)
        if framework == "scikit_learn":
            return self._get_scikit_learn_model_config(scenario)
        if framework == "xgboost":
            return self._get_xgboost_model_config(scenario)
        raise ValueError(f"unsupported framework: {framework}")

    def _get_scikit_learn_model_config(self, scenario: dict):
        scikit_learn = scenario.get("scikit_learn")
        if not isinstance(scikit_learn, dict):
            raise ValueError("scenario is missing scikit_learn block")

        model = scikit_learn.get("model")
        if not isinstance(model, dict):
            raise ValueError("scenario scikit_learn block is missing model")

        if model.get("kind") != "logistic_regression":
            raise ValueError(f"unsupported scikit_learn model kind: {model.get('kind')}")

        return model

    def _get_xgboost_model_config(self, scenario: dict):
        xgboost = scenario.get("xgboost")
        if not isinstance(xgboost, dict):
            raise ValueError("scenario is missing xgboost block")

        model = xgboost.get("model")
        if not isinstance(model, dict):
            raise ValueError("scenario xgboost block is missing model")

        if model.get("kind") != "booster":
            raise ValueError(f"unsupported xgboost model kind: {model.get('kind')}")

        return model
    
    def _get_input_shape(self, scenario: dict):
        model_config = self._get_model_config(scenario)
        layers = model_config['config']['layers']
        if not layers:
            raise ValueError("scenario model does not define any layers")

        input_layer = layers[0]
        input_config = input_layer.get('config', {})
        input_shape = input_config.get('batch_shape')
        if input_shape is None:
            input_shape = input_config.get('batch_input_shape')
        if input_shape is None:
            raise ValueError("scenario model input layer is missing batch_shape")

        return input_shape


    def _get_index_columns(self, parquet_file: pq.ParquetFile) -> set[str]:
        schema = parquet_file.schema_arrow
        metadata = schema.metadata or {}
        pandas_metadata = metadata.get(b'pandas')
        if pandas_metadata is None:
            return {name for name in schema.names if name.startswith("__index_level_")}

        try:
            pandas_config = json.loads(pandas_metadata.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {name for name in schema.names if name.startswith("__index_level_")}

        index_columns: set[str] = set()
        for entry in pandas_config.get("index_columns", []):
            if isinstance(entry, str):
                index_columns.add(entry)
                continue

            if isinstance(entry, dict):
                field_name = entry.get("field_name")
                if field_name:
                    index_columns.add(field_name)

        index_columns.update(
            name for name in schema.names
            if name.startswith("__index_level_")
        )

        return index_columns


    def _get_feature_columns(self, parquet_path: str, label_column: str) -> list[str]:
        parquet_file = pq.ParquetFile(parquet_path)
        schema = parquet_file.schema_arrow
        if label_column not in schema.names:
            raise ValueError(f"label_column ({label_column}) does not match any column in dataset")

        index_columns = self._get_index_columns(parquet_file)
        return [
            name for name in schema.names
            if name != label_column and name not in index_columns
        ]

    
    def validate(self, scenario: dict):
        self.logger.info("Validating scenario for training")
        framework = self._get_framework(scenario)

        # Use parquet metadata instead of reading full dataframes.
        label_column = scenario['label_column']
        parquet_path = os.path.join("share", "test.parquet")
        feature_columns = self._get_feature_columns(parquet_path, label_column)
        num_columns = len(feature_columns)
        if framework == "tensorflow":
            input_shape = self._get_input_shape(scenario)
            if len(input_shape) < 2:
                raise ValueError(f"Model input shape ({input_shape}) is invalid")

            if num_columns != input_shape[1]:
                raise ValueError(f"Model input shape ({input_shape}) is not compatible with dataset ({num_columns} columns)")

            if len(input_shape) == 3 and input_shape[2] != 1:
                raise ValueError(f"Only single-channel sequence inputs are supported, got {input_shape}")
            return

        if framework == "scikit_learn":
            model_config = self._get_model_config(scenario)
            expected_features = model_config.get("n_features")
            if not isinstance(expected_features, int) or expected_features < 1:
                raise ValueError("scenario scikit_learn model is missing a valid n_features value")
            if num_columns != expected_features:
                raise ValueError(
                    f"Scikit-learn model expects {expected_features} feature columns but dataset has {num_columns}"
                )
            classes = model_config.get("classes")
            if not isinstance(classes, list) or len(classes) < 2:
                raise ValueError("scenario scikit_learn model must define at least two classes")
            return

        if framework == "xgboost":
            model_config = self._get_model_config(scenario)
            expected_features = model_config.get("n_features")
            if not isinstance(expected_features, int) or expected_features < 1:
                raise ValueError("scenario xgboost model is missing a valid n_features value")
            if num_columns != expected_features:
                raise ValueError(
                    f"XGBoost model expects {expected_features} feature columns but dataset has {num_columns}"
                )
            return

        raise ValueError(f"unsupported framework: {framework}")
        
