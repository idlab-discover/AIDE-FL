from confluent_kafka import Consumer, Producer

from configuration.config import Config

class KafkaManager:
    def __init__(self, config: Config):
        self.config = config

    def create_consumer(self, topic: str):
        consumer = Consumer({
            'bootstrap.servers': self.config.kafka_address,
            'group.id': self.config.kafka_group,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        })
        consumer.subscribe([topic])
        return consumer
    
    def create_producer(self):
        return Producer({
            'bootstrap.servers': self.config.kafka_address
        })
    
