from configuration.logger import Logger
from service.local_service import LocalService
from service.minio_service import MinioService
from service.kafka_service import KafkaService
from service.preprocessor_service import PreprocessorService

class PreprocessorController:
    def __init__(self, local_service: LocalService, minio_service: MinioService, kafka_service: KafkaService, preprocessor_service: PreprocessorService):
        self.logger = Logger.get_logger("PreprocessorController")
        self.local_service = local_service
        self.minio_service = minio_service
        self.kafka_service = kafka_service
        self.preprocessor_service = preprocessor_service
    
    def StartConsumingPreprocessScenario(self):
        self.logger.info("Consuming scenario.preprocess...")
        self.kafka_service.consume_preprocess_scenario(self.process)

    def process(self, scenario: dict):
        self.logger.info("Start processing")

        # Downloads dataset to ./dataset/dataset.parquet
        dataset_path = self.minio_service.download_dataset(scenario["dataset"])

        # Preprocess
        df_train, df_eval, df_test = self.preprocessor_service.preprocess(scenario["preprocessing"], dataset_path)

        # Remove complete dataset
        self.local_service.delete(dataset_path)
        
        # Save as new parquet files
        train_path = f"{scenario["name"]}_train.parquet"
        eval_path = f"{scenario["name"]}_eval.parquet"
        test_path = f"{scenario["name"]}_test.parquet"
        self.local_service.df_to_parquet(df_train, train_path)
        self.local_service.df_to_parquet(df_eval, eval_path)
        self.local_service.df_to_parquet(df_test, test_path)
        
        # Upload
        self.minio_service.upload_split_dataset(train_path, eval_path, test_path, scenario["dataset"], scenario["name"])
        
        # Remove split dataset
        self.local_service.delete(train_path)
        self.local_service.delete(eval_path)
        self.local_service.delete(test_path)
        
        # Publish scenario preprocessed
        self.kafka_service.produce_preprocessed_scenario(scenario)
