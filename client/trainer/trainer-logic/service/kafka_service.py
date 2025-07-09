import json
from repository.kafka_repository import KafkaConsumerRepository, KafkaProducerRepository

from configuration.logger import Logger
from configuration.messenger import send_error

class KafkaService:
    def __init__(self, train_consumer_repo: KafkaConsumerRepository, trained_producer_repo: KafkaProducerRepository, general_message_producer_repo: KafkaProducerRepository):
        self.logger = Logger.get_logger("KafkaService")
        self.train_consumer = train_consumer_repo
        self.trained_producer = trained_producer_repo
        self.general_message_producer = general_message_producer_repo
    
    def consume_train_scenario(self, handler):
        while True:
            msg = self.train_consumer.poll()
            if msg:
                self.logger.info(f"Received train scenario: {msg.key().decode()}")
                
                scenario_json = msg.value().decode()
                scenario_dict = json.loads(scenario_json)
                
                try:
                    handler(scenario_dict)
                except Exception as e:
                    error_msg = f"Failure in handler: {str(e)}"
                    self.logger.error(error_msg)
                    send_error(error_msg)
                    continue
                finally:
                    self.train_consumer.commit(msg)
                
    def produce_trained_scenario(self, scenario: dict):        
        self.trained_producer.produce(scenario['name'], json.dumps(scenario))
        
    def produce_general_message(self, general_message: dict):
        self.general_message_producer.produce(general_message['type'], json.dumps(general_message))