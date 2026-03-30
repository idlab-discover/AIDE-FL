import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from service.trainer_service import TrainerService
from configuration.logger import Logger


def make_scenario(input_shape):
    return {
        "label_column": "label",
        "model": {
            "config": {
                "layers": [
                    {
                        "config": {
                            "batch_shape": input_shape,
                        }
                    }
                ]
            }
        },
    }


class TrainerServiceTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.cwd = os.getcwd()
        os.chdir(self.temp_dir.name)
        os.makedirs("share", exist_ok=True)
        Logger._config = type("Config", (), {"log_level": "INFO"})()
        Logger._loggers = {}
        self.service = TrainerService()

    def tearDown(self):
        os.chdir(self.cwd)
        self.temp_dir.cleanup()

    def write_test_parquet(self, frame: pd.DataFrame):
        frame.to_parquet(Path("share") / "test.parquet")

    def test_validate_ignores_named_index_columns(self):
        frame = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0],
                "feature_b": [3.0, 4.0],
                "label": [0.0, 1.0],
            },
            index=pd.Index([10, 11], name="row_id"),
        )
        self.write_test_parquet(frame)

        self.service.validate(make_scenario([None, 2]))

    def test_validate_supports_single_channel_sequence_inputs(self):
        frame = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0],
                "feature_b": [3.0, 4.0],
                "label": [0.0, 1.0],
            }
        )
        self.write_test_parquet(frame)

        self.service.validate(make_scenario([None, 2, 1]))

    def test_validate_rejects_feature_count_mismatch(self):
        frame = pd.DataFrame(
            {
                "feature_a": [1.0, 2.0],
                "feature_b": [3.0, 4.0],
                "label": [0.0, 1.0],
            }
        )
        self.write_test_parquet(frame)

        with self.assertRaisesRegex(ValueError, "not compatible with dataset"):
            self.service.validate(make_scenario([None, 3]))


if __name__ == "__main__":
    unittest.main()
