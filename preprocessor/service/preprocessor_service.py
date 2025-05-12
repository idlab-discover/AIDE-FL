from sklearn.model_selection import train_test_split
import pandas as pd

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
    

    def shuffle(self, df: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Shuffling")
        return df.sample(frac=1)
    

    def split(self, df: pd.DataFrame, train: float, eval: float, test: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_train, df_temp = train_test_split(df, test_size=(1 - train), shuffle=False)
        eval_ratio = eval / (eval + test)
        df_eval, df_test = train_test_split(df_temp, test_size=(1 - eval_ratio), shuffle=False)

        self.logger.info(f"Split dataset: train={len(df_train)}, eval={len(df_eval)}, test={len(df_test)}")
        return df_train, df_eval, df_test