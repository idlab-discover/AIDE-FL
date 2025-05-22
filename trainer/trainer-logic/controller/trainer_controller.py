from configuration.logger import Logger
from service.local_service import LocalService
from service.minio_service import MinioService
from service.kafka_service import KafkaService
from service.trainer_service import TrainerService

class TrainerController:
    def __init__(self, local_service: LocalService, minio_service: MinioService, kafka_service: KafkaService, trainer_service: TrainerService):
        self.logger = Logger.get_logger("PreprocessorController")
        self.local_service = local_service
        self.minio_service = minio_service
        self.kafka_service = kafka_service
        self.trainer_service = trainer_service
    
    
    def StartConsumingTrainScenario(self):
        self.logger.info("Consuming scenario.train...")
        self.kafka_service.consume_train_scenario(self.process)


    def process(self, scenario: dict):
        self.logger.info("Start processing")

        # Download datasets (to shared volume)
        self.minio_service.download_split_datasets(scenario["dataset"], scenario["name"])

        # Set config for Flower clientapp
        self.trainer_service.set_flower_config(scenario)

        # Publish scenario trained
        self.kafka_service.produce_trained_scenario(scenario)
