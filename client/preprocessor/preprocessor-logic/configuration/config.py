from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    def __init__(self):
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        self.minio_datasets_bucket = os.getenv("MINIO_DATASETS_BUCKET")
        self.minio_server_endpoint = os.getenv("MINIO_SERVER_ENDPOINT")
        self.minio_server_access_key = os.getenv("MINIO_SERVER_ACCESS_KEY")
        self.minio_server_secret_key = os.getenv("MINIO_SERVER_SECRET_KEY")
        self.minio_server_datasets_bucket = os.getenv("MINIO_SERVER_DATASETS_BUCKET")
        self.redis_address = os.getenv("REDIS_ADDRESS")
        self.redis_exp = os.getenv("REDIS_EXP")
        self.kafka_address = os.getenv("KAFKA_ADDRESS")
        self.kafka_group = os.getenv("KAFKA_GROUP")
        self.kafka_preprocess_topic = os.getenv("KAFKA_PREPROCESS_TOPIC")
        self.kafka_preprocessed_topic = os.getenv("KAFKA_PREPROCESSED_TOPIC")
        self.kafka_message_topic = os.getenv("KAFKA_MESSAGE_TOPIC")
        self.log_level = os.getenv("LOG_LEVEL")