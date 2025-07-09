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
        num_columns = len(schema) - 2 # Minus one for label column and minus another for some non existing name added by pyarrow (__index_level_0__)
        input_layer = scenario['model']['config']['layers'][0]
        input_shape = input_layer['config']['batch_shape']
        if num_columns != input_shape[1]:
            raise ValueError(f"Model input shape ({input_shape}) is not compatible with dataset ({num_columns} columns)")
        