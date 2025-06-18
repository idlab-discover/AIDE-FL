from configuration.logger import Logger
from service.local_service import LocalService
from service.minio_service import MinioService
from service.redis_service import RedisService
from service.kafka_service import KafkaService
from service.preprocessor_service import PreprocessorService

class PreprocessorController:
    def __init__(self, local_service: LocalService, minio_service: MinioService, redis_service: RedisService, kafka_service: KafkaService, preprocessor_service: PreprocessorService):
        self.logger = Logger.get_logger("PreprocessorController")
        self.local_service = local_service
        self.minio_service = minio_service
        self.redis_service = redis_service
        self.kafka_service = kafka_service
        self.preprocessor_service = preprocessor_service
    
    def StartConsumingPreprocessScenario(self):
        self.logger.info("Consuming scenario.preprocess...")
        self.kafka_service.consume_preprocess_scenario(self.process)

    def process(self, scenario: dict):
        self.logger.info("Start processing")
        
        # Check whether preprocessed dataset is cached in Redis
        if not self.redis_service.are_datasets_cached(scenario["name"]):
            
            # Downloads dataset to ./dataset/dataset.parquet
            dataset_path = self.minio_service.download_dataset(scenario["dataset"])

            # Preprocess
            df_train, df_eval, df_test = self.preprocessor_service.preprocess(scenario["preprocessing"], dataset_path)

            # Remove complete dataset
            self.local_service.delete(dataset_path)
            
            # Upload
            self.redis_service.store_dataframes_as_parquet(scenario["name"], df_train, df_eval, df_test)
            
        else:
            # Reset TTL of datasets
            self.redis_service.reset_datasets_ttl(scenario["name"])
            
        # Publish scenario preprocessed
        self.kafka_service.produce_preprocessed_scenario(scenario)
