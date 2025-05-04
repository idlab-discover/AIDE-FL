from configuration.config import Config
from configuration.minio import MinioClient
from configuration.kafka import KafkaManager
from configuration.logger import Logger
from repository.minio_repository import MinioRepository
from repository.kafka_repository import KafkaConsumerRepository, KafkaProducerRepository
from service.minio_service import MinioService
from service.kafka_service import KafkaService
from controller.preprocessor_controller import PreprocessorController

def main():
    # setup configuration
    config = Config()
    minio_client = MinioClient(config)
    kafka_manager = KafkaManager(config)
    kafka_preprocess_consumer = kafka_manager.create_consumer(config.kafka_preprocess_topic)
    kafka_producer = kafka_manager.create_producer()
    Logger.init(config)
    
    # repository
    minio_repository = MinioRepository(minio_client, config)
    kafka_preprocess_consumer_repository = KafkaConsumerRepository(kafka_preprocess_consumer)
    kafka_producer_repository = KafkaProducerRepository(kafka_producer)
    
    # service
    minio_service = MinioService(minio_repository)
    kafka_service = KafkaService(kafka_preprocess_consumer_repository, kafka_producer_repository)

    # controller
    preprocessor_controller = PreprocessorController(minio_service, kafka_service)
    
    # consume
    preprocessor_controller.StartConsumingPreprocessScenario()

if __name__ == "__main__":
    main()