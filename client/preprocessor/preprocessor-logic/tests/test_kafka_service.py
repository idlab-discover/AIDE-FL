import json
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

dotenv_module = ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

confluent_kafka_module = ModuleType("confluent_kafka")
confluent_kafka_module.Consumer = object
confluent_kafka_module.Producer = object
sys.modules.setdefault("confluent_kafka", confluent_kafka_module)

from configuration.messenger import Messenger
from configuration.logger import Logger
from service.kafka_service import KafkaService


class StubMessage:
    def __init__(self, key, value):
        self._key = key
        self._value = value

    def key(self):
        return self._key

    def value(self):
        return self._value


class StubConsumer:
    def __init__(self):
        self.committed = []

    def commit(self, msg):
        self.committed.append(msg)


class StubProducer:
    def produce(self, *_args, **_kwargs):
        pass


class KafkaServiceTests(unittest.TestCase):
    def setUp(self):
        Messenger._instance = None
        Logger._config = SimpleNamespace(log_level="INFO")
        Logger._loggers = {}
        Messenger.init(lambda _message: None)
        self.consumer = StubConsumer()
        self.service = KafkaService(self.consumer, StubProducer(), StubProducer())

    def test_process_message_commits_after_success(self):
        msg = StubMessage(b"prep-1", json.dumps({"prep_id": "prep-1"}).encode())

        seen = []
        self.service._process_message(msg, lambda scenario: seen.append(scenario["prep_id"]))

        self.assertEqual(seen, ["prep-1"])
        self.assertEqual(self.consumer.committed, [msg])
        self.assertIsNone(Messenger.get_messenger()._prep_id)

    @patch("service.kafka_service.send_error")
    def test_process_message_commits_after_error_message(self, send_error):
        msg = StubMessage(b"prep-1", json.dumps({"prep_id": "prep-1"}).encode())

        def handler(_scenario):
            raise RuntimeError("boom")

        self.service._process_message(msg, handler)

        send_error.assert_called_once_with("Failure in handler: boom")
        self.assertEqual(self.consumer.committed, [msg])
        self.assertIsNone(Messenger.get_messenger()._prep_id)

    @patch("service.kafka_service.send_error", side_effect=RuntimeError("send failed"))
    def test_process_message_does_not_commit_when_error_publish_fails(self, _send_error):
        msg = StubMessage(b"prep-1", json.dumps({"prep_id": "prep-1"}).encode())

        def handler(_scenario):
            raise RuntimeError("boom")

        with self.assertRaisesRegex(RuntimeError, "send failed"):
            self.service._process_message(msg, handler)

        self.assertEqual(self.consumer.committed, [])
        self.assertIsNone(Messenger.get_messenger()._prep_id)


if __name__ == "__main__":
    unittest.main()
