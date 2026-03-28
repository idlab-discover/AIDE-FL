from configuration.logger import Logger

import pyarrow.parquet as pq
import json
import os

class TrainerService:
    def __init__(self):
        self.logger = Logger.get_logger("TrainerService")
    
    
    def validate(self, scenario: dict):
        self.logger.info("Validating scenario for training")

        # Use pyarrow instead of pandas as it is faster for metadata-only checks
        schema = pq.ParquetFile(os.path.join("share", "test.parquet")).schema_arrow
        
        # Check if label column exists in dataset (suppose that label_column property is present in the scenario as it has been checked by server)
        label_column = scenario['label_column']
        if label_column not in schema.names:
            raise ValueError(f"label_column ({label_column}) does not match any column in dataset")
        
        # Check if scenario model is compatible with dataset's shape
        feature_columns = [
            name for name in schema.names
            if name != label_column and not name.startswith("__index_level_")
        ]
        num_columns = len(feature_columns)
        input_layer = scenario['model']['config']['layers'][0]
        input_shape = input_layer['config']['batch_shape']
        if num_columns != input_shape[1]:
            raise ValueError(f"Model input shape ({input_shape}) is not compatible with dataset ({num_columns} columns)")
        
