#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

DEFAULT_FRAMEWORK = "tensorflow"
DEFAULT_AGGREGATION_STRATEGY = "fedavg"


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def warn_deprecated(field: str, replacement: str) -> None:
    sys.stderr.write(
        f"warning: scenario field '{field}' is deprecated; use '{replacement}' instead\n"
    )


def resolve_json_path(base_path: Path, candidate: str) -> Path:
    resolved = Path(candidate)
    if resolved.is_absolute():
        return resolved
    return base_path.parent / resolved


def normalize_training_and_aggregation(scenario: dict) -> None:
    legacy_hyper = scenario.pop("hyper", None)
    if legacy_hyper is not None:
        warn_deprecated("hyper", "training")
        scenario["training"] = legacy_hyper

    legacy_strategy = scenario.pop("strategy", None)
    if legacy_strategy is not None:
        warn_deprecated("strategy", "aggregation")
        scenario["aggregation"] = {
            "strategy": DEFAULT_AGGREGATION_STRATEGY,
            **legacy_strategy,
        }


def normalize_tensorflow_block(scenario: dict, scenario_path: Path) -> None:
    if scenario.get("framework") != DEFAULT_FRAMEWORK:
        return

    tensorflow = dict(scenario.get("tensorflow") or {})
    tensorflow_model = dict(tensorflow.get("model") or {})

    legacy_model = scenario.pop("model", None)
    if legacy_model is not None:
        warn_deprecated("model", "tensorflow.model")
        tensorflow_model["config"] = legacy_model

    legacy_model_file = scenario.pop("model_file", None)
    if legacy_model_file is not None:
        warn_deprecated("model_file", "tensorflow.model.config_file")
        tensorflow_model["config_file"] = legacy_model_file

    if "config" not in tensorflow_model and "config_file" in tensorflow_model:
        model_path = resolve_json_path(scenario_path, tensorflow_model["config_file"])
        tensorflow_model["config"] = load_json(model_path)
        tensorflow_model.pop("config_file", None)

    if tensorflow_model:
        tensorflow["model"] = tensorflow_model
        scenario["tensorflow"] = tensorflow


def render_scenario(path: Path) -> dict:
    scenario = load_json(path)
    normalize_training_and_aggregation(scenario)
    normalize_tensorflow_block(scenario, path)
    return scenario


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a scenario by expanding authored references into inline config."
    )
    parser.add_argument("scenario", help="Path to the authored scenario JSON file")
    parser.add_argument("-o", "--output", help="Optional output file path. Defaults to stdout.")
    args = parser.parse_args()

    scenario_path = Path(args.scenario).resolve()
    rendered = render_scenario(scenario_path)

    output = json.dumps(rendered, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        sys.stdout.write(output)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
