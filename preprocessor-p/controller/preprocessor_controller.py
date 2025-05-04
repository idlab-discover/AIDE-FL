from configuration.logger import Logger
from service.minio_service import MinioService
from service.kafka_service import KafkaService

class PreprocessorController:
    def __init__(self, minio_service: MinioService, kafka_service: KafkaService):
        self.logger = Logger.get_logger("PreprocessorController")
        self.minio_service = minio_service
        self.kafka_service = kafka_service
    
    def StartConsumingPreprocessScenario(self):
        self.logger.info("Consuming scenario.preprocess...")
        self.kafka_service.consume_preprocess_scenario(self.preprocess)

    def preprocess(self, scenario: str):
        self.logger.info("Start preprocessing")
        self.logger.info(scenario)