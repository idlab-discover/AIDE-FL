import json
import tempfile
import unittest
from pathlib import Path

from render_scenario import render_scenario


class RenderScenarioTest(unittest.TestCase):
    def test_model_file_is_expanded_relative_to_scenario(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "models").mkdir()

            model_path = temp_path / "models" / "demo-model.json"
            model_path.write_text('{"class_name":"Sequential"}', encoding="utf-8")

            scenario_path = temp_path / "scenario.json"
            scenario_path.write_text(
                json.dumps(
                    {
                        "name": "demo",
                        "dataset": "dataset",
                        "label_column": "label",
                        "preprocessing": {
                            "drop_columns": [],
                            "drop_rows": 0,
                            "shuffle": False,
                            "split": {"train": 0.7, "eval": 0.2, "test": 0.1},
                        },
                        "hyper": {"server_rounds": 1, "local_epochs": 1, "batch_size": 16},
                        "strategy": {"fraction_fit": 1, "fraction_evaluate": 1, "min_available_clients": 1},
                        "model_file": "models/demo-model.json",
                    }
                ),
                encoding="utf-8",
            )

            rendered = render_scenario(scenario_path)

            self.assertEqual(rendered["model_file"], "models/demo-model.json")
            self.assertEqual(rendered["model"]["class_name"], "Sequential")


if __name__ == "__main__":
    unittest.main()
