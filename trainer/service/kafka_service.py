import json
from repository.kafka_repository import KafkaConsumerRepository, KafkaProducerRepository

from configuration.logger import Logger

class KafkaService:
    def __init__(self, train_consumer_repo: KafkaConsumerRepository, trained_producer_repo: KafkaProducerRepository):
        self.logger = Logger.get_logger("KafkaService")
        self.train_consumer = train_consumer_repo
        self.trained_producer = trained_producer_repo
    
    def consume_train_scenario(self, handler):
        while True:
            msg = self.train_consumer.poll()
            if msg:
                self.logger.info(f"Received train scenario: {msg.key().decode()}")
                
                scenario_json = msg.value().decode()
                scenario_dict = json.loads(scenario_json)
                handler(scenario_dict)

                self.train_consumer.commit(msg)
                
                
    def produce_trained_scenario(self, scenario: dict):        
        self.trained_producer.produce(scenario['name'], json.dumps(scenario))