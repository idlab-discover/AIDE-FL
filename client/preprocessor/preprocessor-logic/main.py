from configuration.config import Config
from configuration.minio import MinioClient
from configuration.kafka import KafkaFactory
from configuration.logger import Logger
from repository.minio_repository import MinioRepository
from repository.kafka_repository import KafkaConsumerRepository, KafkaProducerRepository
from service.local_service import LocalService
from service.minio_service import MinioService
from service.kafka_service import KafkaService
from service.preprocessor_service import PreprocessorService
from controller.preprocessor_controller import PreprocessorController

def main():
    # setup configuration
    config = Config()
    Logger.init(config)
    minio_client = MinioClient(config)
    kafka_factory = KafkaFactory(config)
    
    # repository
    minio_repository = MinioRepository(minio_client, config)
    kafka_preprocess_consumer_repository = kafka_factory.new_consumer_repository(config.kafka_preprocess_topic)
    kafka_preprocessed_producer_repository = kafka_factory.new_producer_repository(config.kafka_preprocessed_topic)
    
    # service
    local_service = LocalService("/")
    minio_service = MinioService(minio_repository)
    kafka_service = KafkaService(kafka_preprocess_consumer_repository, kafka_preprocessed_producer_repository)
    preprocessor_service = PreprocessorService()

    # controller
    preprocessor_controller = PreprocessorController(local_service, minio_service, kafka_service, preprocessor_service)
    
    # consume
    preprocessor_controller.StartConsumingPreprocessScenario()

if __name__ == "__main__":
    main()