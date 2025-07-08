from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from configuration.logger import Logger

class PreprocessorService:
    def __init__(self):
        self.logger = Logger.get_logger("PreprocessorService")


    def preprocess(self, preprocessing_steps: dict, parquet_filename: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.logger.info("Start preprocessing")

        df = pd.read_parquet(parquet_filename)

        # Drop columns
        if "drop_columns" in preprocessing_steps:
            df = self.drop_columns(df, preprocessing_steps["drop_columns"])
            
        # Drop rows
        if "drop_rows" in preprocessing_steps:
            df = self.drop_rows(df, preprocessing_steps["drop_rows"])

        # Shuffle
        if "shuffle" in preprocessing_steps and preprocessing_steps["shuffle"] is True:
            df = self.shuffle(df)

        # Train, eval and test split
        split_conf = preprocessing_steps["split"]
        train_split_conf = split_conf["train"]
        eval_split_conf = split_conf["eval"]
        test_split_conf = split_conf["test"]
        df_train, df_eval, df_test = self.split(df, train_split_conf, eval_split_conf, test_split_conf)
        
        return df_train, df_eval, df_test


    def drop_columns(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        self.logger.info(f"Dropping columns: {cols}")
        return df.drop(columns=cols)
    
    
    def drop_rows(self, df: pd.DataFrame, n: int) -> pd.DataFrame:
        if n >= len(df):
            raise ValueError(f"Requested amount of rows to drop ({n}) is >= dataset size {len(df)}")
        
        self.logger.info(f"Dropping {n} rows out of {len(df)}")
        drop_indices = np.random.choice(df.index, n, replace=False)
        df_subset = df.drop(drop_indices)
        return df_subset
    

    def shuffle(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Shuffling")
        return df.sample(frac=1)
    

    def split(self, df: pd.DataFrame, train: float, eval: float, test: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if train + eval + test != 1.0:
            raise ValueError(f"Sum of train, validation, and test splits should be 1.0, not {train + eval + test}")
        
        df_train, df_temp = train_test_split(df, test_size=(1 - train), shuffle=False)
        eval_ratio = eval / (eval + test)
        df_eval, df_test = train_test_split(df_temp, test_size=(1 - eval_ratio), shuffle=False)

        self.logger.info(f"Split dataset: train={len(df_train)}, eval={len(df_eval)}, test={len(df_test)}")
        return df_train, df_eval, df_test