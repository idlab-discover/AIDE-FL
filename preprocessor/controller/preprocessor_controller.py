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
        self.preprocessor_service.preprocess(scenario["preprocessing"], dataset_path)

        # Remove complete dataset
        self.local_service.delete(dataset_path)
        
        # Publish scenario preprocessed
        self.kafka_service.produce_preprocessed_scenario(scenario)
