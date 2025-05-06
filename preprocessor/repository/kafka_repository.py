from configuration.logger import Logger

from confluent_kafka import Consumer, Producer

class KafkaConsumerRepository:
    def __init__(self, consumer: Consumer):
        self.logger = Logger.get_logger("KafkaConsumerRepository")
        self.consumer = consumer

    def poll(self):
        msg = self.consumer.poll(1.0)
        if msg and not msg.error():
            return msg
        return None
    
    def commit(self, msg):
        self.logger.info(f"Committing msg {msg.key().decode()}")
        self.consumer.commit(msg)
    
    def close(self):
        self.logger.info("Closing consumer")
        self.consumer.close()

class KafkaProducerRepository:
    def __init__(self, producer: Producer):
        self.logger = Logger.get_logger("KafkaProducerRepository")
        self.producer = producer

    def produce(self, topic: str, key: str, value: str):
        self.logger.info(f"Producing {key} to {topic}")
        self.producer.produce(topic, key=key, value=value)
        self.producer.flush()