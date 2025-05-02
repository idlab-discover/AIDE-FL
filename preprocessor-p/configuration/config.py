from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    def __init__(self):
        self.minio_endpoint = os.getenv("MINIO_ENDPOINT")
        self.minio_access_key = os.getenv("MINIO_ACCESS_KEY")
        self.minio_secret_key = os.getenv("MINIO_SECRET_KEY")
        self.minio_bucket = os.getenv("MINIO_BUCKET")
        self.kafka_address = os.getenv("KAFKA_ADDRESS")
        self.kafka_group = os.getenv("KAFKA_GROUP")
        self.kafka_preprocess_topic = os.getenv("KAFKA_PREPROCESS_TOPIC")
        self.kafka_preprocessed_topic = os.getenv("KAFKA_PREPROCESSED_TOPIC")