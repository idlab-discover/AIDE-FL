from confluent_kafka import Consumer, Producer

from configuration.config import Config
from repository.kafka_repository import KafkaConsumerRepository, KafkaProducerRepository

class KafkaFactory:
    def __init__(self, config: Config):
        self.config = config
        
    
    def new_consumer_repository(self, topic: str) -> KafkaConsumerRepository:
        consumer = Consumer({
            'bootstrap.servers': self.config.kafka_address,
            'group.id': self.config.kafka_group,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        })
        consumer.subscribe([topic])
        return KafkaConsumerRepository(consumer)
    
    
    def new_producer_repository(self, topic: str) -> KafkaProducerRepository:
        producer = Producer({
            'bootstrap.servers': self.config.kafka_address
        })
        return KafkaProducerRepository(producer, topic)
