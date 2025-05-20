from configuration.config import Config
from configuration.minio import MinioClient
from configuration.kafka import KafkaFactory
from configuration.logger import Logger
from repository.minio_repository import MinioRepository
from service.local_service import LocalService
from service.minio_service import MinioService
from service.kafka_service import KafkaService
from trainer.service.trainer_service import TrainerService
from trainer.controller.trainer_controller import TrainerController

def main():
    # setup configuration
    config = Config()
    Logger.init(config)
    minio_client = MinioClient(config)
    kafka_factory = KafkaFactory(config)
    
    # repository
    minio_repository = MinioRepository(minio_client, config)
    kafka_train_consumer_repository = kafka_factory.new_consumer_repository(config.kafka_train_topic)
    kafka_trained_producer_repository = kafka_factory.new_producer_repository(config.kafka_trained_topic)
    
    # service
    local_service = LocalService("/")
    minio_service = MinioService(minio_repository)
    kafka_service = KafkaService(kafka_train_consumer_repository, kafka_trained_producer_repository)
    trainer_service = TrainerService()

    # controller
    preprocessor_controller = TrainerController(local_service, minio_service, kafka_service, trainer_service)
    
    # consume
    preprocessor_controller.StartConsumingTrainScenario()

if __name__ == "__main__":
    main()