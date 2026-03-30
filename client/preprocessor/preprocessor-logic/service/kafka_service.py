import json
from repository.kafka_repository import KafkaConsumerRepository, KafkaProducerRepository

from configuration.logger import Logger
from configuration.messenger import Messenger, send_error

class KafkaService:
    def __init__(self, preprocess_consumer_repo: KafkaConsumerRepository, preprocessed_producer_repo: KafkaProducerRepository, general_message_producer_repo: KafkaProducerRepository):
        self.logger = Logger.get_logger("KafkaService")
        self.preprocess_consumer = preprocess_consumer_repo
        self.preprocessed_producer = preprocessed_producer_repo
        self.general_message_producer = general_message_producer_repo
    
    def consume_preprocess_scenario(self, handler):
        self.logger.info("Started consuming preprocess scenarios")
        
        while True:
            msg = self.preprocess_consumer.poll()
            if msg:
                self._process_message(msg, handler)

    def _process_message(self, msg, handler):
        key = msg.key().decode() if msg.key() is not None else "<nil>"
        self.logger.info(f"Received preprocess scenario: {key}")

        should_commit = False

        try:
            scenario_json = msg.value().decode()
            scenario_dict = json.loads(scenario_json)
            Messenger.set_prep_id(scenario_dict.get("prep_id"))
            handler(scenario_dict)
            should_commit = True
        except Exception as e:
            error_msg = f"Failure in handler: {str(e)}"
            self.logger.error(error_msg)
            send_error(error_msg)
            should_commit = True
        finally:
            Messenger.clear_prep_id()
            if should_commit:
                self.preprocess_consumer.commit(msg)
                     
    def produce_preprocessed_scenario(self, scenario: dict):        
        self.preprocessed_producer.produce(scenario['prep_id'], json.dumps(scenario))
        
    def produce_general_message(self, general_message: dict):
        key = general_message.get('prep_id', general_message['type'])
        self.general_message_producer.produce(key, json.dumps(general_message))
