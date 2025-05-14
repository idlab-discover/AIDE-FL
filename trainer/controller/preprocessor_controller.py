from configuration.logger import Logger
from service.local_service import LocalService
from service.minio_service import MinioService
from service.kafka_service import KafkaService
from service.train_service import TrainService

class PreprocessorController:
    def __init__(self, local_service: LocalService, minio_service: MinioService, kafka_service: KafkaService, train_service: TrainService):
        self.logger = Logger.get_logger("PreprocessorController")
        self.local_service = local_service
        self.minio_service = minio_service
        self.kafka_service = kafka_service
        self.train_service = train_service
    
    
    def StartConsumingTrainScenario(self):
        self.logger.info("Consuming scenario.train...")
        self.kafka_service.consume_train_scenario(self.process)


    def process(self, scenario: dict):
        self.logger.info("Start processing")

        # Publish scenario preprocessed
        self.kafka_service.produce_trained_scenario(scenario)
