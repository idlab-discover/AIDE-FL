from confluent_kafka import Consumer, Producer

from configuration.logger import Logger

class KafkaService:
    def __init__(self, preprocess_consumer_repo: Consumer, producer_repo: Producer):
        self.logger = Logger.get_logger("KafkaService")
        self.preprocess_consumer = preprocess_consumer_repo
        self.producer = producer_repo
    
    def consume_preprocess_scenario(self, handler):
        while True:
            msg = self.preprocess_consumer.poll()
            if msg:
                self.logger.info(f"Received preprocess scenario: {msg.key().decode()}")
                handler(msg.value().decode())
                self.preprocess_consumer.commit(msg)