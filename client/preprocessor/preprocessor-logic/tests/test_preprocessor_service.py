import os
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configuration.logger import Logger
from service.preprocessor_service import PreprocessorService


class PreprocessorServiceTest(unittest.TestCase):
    def setUp(self):
        Logger._config = type("Config", (), {"log_level": "INFO"})()
        Logger._loggers = {}
        self.service = PreprocessorService()
        self.frame = pd.DataFrame({"value": list(range(20))})

    def test_split_accepts_close_sum(self):
        train, eval_, test = self.service.split(self.frame, 0.333333, 0.333333, 0.333334)
        self.assertEqual(len(train) + len(eval_) + len(test), len(self.frame))

    def test_split_supports_zero_eval(self):
        train, eval_, test = self.service.split(self.frame, 0.7, 0.0, 0.3)
        self.assertEqual(len(train), 14)
        self.assertEqual(len(eval_), 0)
        self.assertEqual(len(test), 6)

    def test_split_supports_zero_test(self):
        train, eval_, test = self.service.split(self.frame, 0.7, 0.3, 0.0)
        self.assertEqual(len(train), 14)
        self.assertEqual(len(eval_), 6)
        self.assertEqual(len(test), 0)

    def test_split_rejects_zero_train(self):
        with self.assertRaisesRegex(ValueError, "Train split must be greater than 0"):
            self.service.split(self.frame, 0.0, 0.5, 0.5)

    def test_split_rejects_missing_eval_and_test(self):
        with self.assertRaisesRegex(ValueError, "At least one of eval or test must be greater than 0"):
            self.service.split(self.frame, 1.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
