import json
from repository.kafka_repository import KafkaConsumerRepository, KafkaProducerRepository

from configuration.logger import Logger

class KafkaService:
    def __init__(self, preprocess_consumer_repo: KafkaConsumerRepository, preprocessed_producer_repo: KafkaProducerRepository):
        self.logger = Logger.get_logger("KafkaService")
        self.preprocess_consumer = preprocess_consumer_repo
        self.preprocessed_producer = preprocessed_producer_repo
    
    def consume_preprocess_scenario(self, handler):
        while True:
            msg = self.preprocess_consumer.poll()
            if msg:
                self.logger.info(f"Received preprocess scenario: {msg.key().decode()}")
                
                scenario_json = msg.value().decode()
                scenario_dict = json.loads(scenario_json)
                handler(scenario_dict)

                self.preprocess_consumer.commit(msg)
                
                
    def produce_preprocessed_scenario(self, scenario: dict):        
        self.preprocessed_producer.produce(scenario['name'], json.dumps(scenario))