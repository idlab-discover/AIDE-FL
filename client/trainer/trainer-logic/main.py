from configuration.config import Config
from configuration.kafka import KafkaFactory
from configuration.redis import RedisClient
from configuration.logger import Logger
from configuration.messenger import Messenger
from repository.redis_repository import RedisRepository
from service.local_service import LocalService
from service.redis_service import RedisService
from service.kafka_service import KafkaService
from service.trainer_service import TrainerService
from controller.trainer_controller import TrainerController

def main():
    # setup configuration
    config = Config()
    Logger.init(config)
    kafka_factory = KafkaFactory(config)
    redis_client = RedisClient(config)
    
    # repository
    redis_repository = RedisRepository(redis_client, config)
    kafka_train_consumer_repository = kafka_factory.new_consumer_repository(config.kafka_train_topic)
    kafka_trained_producer_repository = kafka_factory.new_producer_repository(config.kafka_trained_topic)
    kafka_general_message_producer_repository = kafka_factory.new_producer_repository(config.kafka_message_topic)
    
    # service
    local_service = LocalService("/")
    redis_service = RedisService(redis_repository)
    kafka_service = KafkaService(kafka_train_consumer_repository, kafka_trained_producer_repository, kafka_general_message_producer_repository)
    trainer_service = TrainerService()
    
    # Initialise messenger
    Messenger.init(kafka_service.produce_general_message)

    # controller
    trainer_controller = TrainerController(local_service, redis_service, kafka_service, trainer_service)
    
    # consume
    trainer_controller.StartConsumingTrainScenario()

if __name__ == "__main__":
    main()