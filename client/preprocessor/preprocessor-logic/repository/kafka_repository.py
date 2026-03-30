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
        key = msg.key().decode() if msg.key() is not None else "<nil>"
        self.logger.info(f"Committing msg {key}")
        self.consumer.commit(msg)
    
    
    def close(self):
        self.logger.info("Closing consumer")
        self.consumer.close()


    def __del__(self):
        self.close()

class KafkaProducerRepository:
    def __init__(self, producer: Producer, topic: str):
        self.logger = Logger.get_logger("KafkaProducerRepository")
        self.producer = producer
        self.topic = topic


    def produce(self, key: str, value: str):
        self.logger.info(f"Producing {key} to {self.topic}")
        self.producer.produce(self.topic, key=key, value=value)
        self.producer.flush()
