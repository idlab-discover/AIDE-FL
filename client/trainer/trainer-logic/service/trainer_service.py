from configuration.logger import Logger

import pyarrow.parquet as pq
import json
import os

class TrainerService:
    def __init__(self):
        self.logger = Logger.get_logger("TrainerService")
    
    def _get_input_shape(self, scenario: dict):
        layers = scenario['model']['config']['layers']
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

        # Use parquet metadata instead of reading full dataframes.
        label_column = scenario['label_column']
        parquet_path = os.path.join("share", "test.parquet")
        feature_columns = self._get_feature_columns(parquet_path, label_column)
        num_columns = len(feature_columns)
        input_shape = self._get_input_shape(scenario)
        if len(input_shape) < 2:
            raise ValueError(f"Model input shape ({input_shape}) is invalid")

        if num_columns != input_shape[1]:
            raise ValueError(f"Model input shape ({input_shape}) is not compatible with dataset ({num_columns} columns)")

        if len(input_shape) == 3 and input_shape[2] != 1:
            raise ValueError(f"Only single-channel sequence inputs are supported, got {input_shape}")
        
