from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    def __init__(self):
        self.redis_address = os.getenv("REDIS_ADDRESS")
        self.kafka_address = os.getenv("KAFKA_ADDRESS")
        self.kafka_group = os.getenv("KAFKA_GROUP")
        self.kafka_train_topic = os.getenv("KAFKA_TRAIN_TOPIC")
        self.kafka_trained_topic = os.getenv("KAFKA_TRAINED_TOPIC")
        self.log_level = os.getenv("LOG_LEVEL")